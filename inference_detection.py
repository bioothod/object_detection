import argparse
import cv2
import json
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

import numpy as np
import tensorflow as tf

from collections import defaultdict

import anchors_gen
import encoder

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--max_ret', type=int, default=100, help='Maximum number of returned boxes')
parser.add_argument('--min_obj_score', type=float, default=0.3, help='Minimal class probability')
parser.add_argument('--min_size', type=float, default=10, help='Minimal size of the bounding box')
parser.add_argument('--iou_threshold', type=float, default=0.45, help='Minimal IoU threshold for non-maximum suppression')
parser.add_argument('--output_dir', type=str, required=True, help='Path to directory, where images will be stored')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--checkpoint_dir', type=str, help='Load model weights from the latest checkpoint in this directory')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--skip_checkpoint_assertion', action='store_true', help='Skip checkpoint assertion, needed for older models on objdet3/4')
parser.add_argument('--no_channel_swap', action='store_true', help='When set, do not perform rgb-bgr conversion, needed, when using files as input')
parser.add_argument('--freeze', action='store_true', help='Save frozen protobuf near checkpoint')
parser.add_argument('--do_not_save_images', action='store_true', help='Do not save images with bounding boxes')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--save_crops_dir', type=str, help='Directory to save slightly upscaled text crops')
parser.add_argument('--dataset_type', type=str, choices=['files', 'tfrecords', 'filelist'], default='files', help='Dataset type')
parser.add_argument('--image_size', type=int, required=True, help='Use this image size, if 0 - use default')
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')

default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
parser.add_argument('--dictionary', type=str, default=default_char_dictionary, help='Dictionary to use')
FLAGS = parser.parse_args()

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2, tf.int32),
                tf.cast((mx - orig_image_width) / 2, tf.int32),
                mx_int,
                mx_int)

    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, dtype)

    image -= 128
    image /= 128

    return filename, image

def tf_left_needed_dimensions_from_tfrecord(image_size, anchors_all, filename, image, true_values, dictionary_size):
    char_boundary_start = 0
    word_boundary_start = dictionary_size + 1 + 4 * 2

    true_char = true_values[..., char_boundary_start : char_boundary_start + word_boundary_start]
    true_word = true_values[..., word_boundary_start : ]

    true_char_obj = true_char[..., 0]
    true_char_poly = true_char[..., 1 : 9]
    true_char_letters = true_char[..., 10 :]

    true_word_obj = true_word[..., 0]
    true_word_poly = true_word[..., 1 : 9]

    char_index = tf.where(true_char_obj != 0)
    word_index = tf.where(true_word_obj != 0)

    char_poly = tf.gather(true_char_poly, char_index)
    char_poly = tf.reshape(char_poly, [-1, 4, 2])
    word_poly = tf.gather(true_word_poly, word_index).numpy()
    word_poly = tf.reshape(word_poly, [-1, 4, 2])

    best_anchors = tf.gather(all_anchors[..., :2], char_index)
    best_anchors = tf.expand_dims(best_anchors, 1)
    best_anchors = tf.tile(best_anchors, [1, 4, 1])
    char_poly = char_poly + best_anchors

    best_anchors = tf.gather(all_anchors[..., :2], word_index)
    best_anchors = tf.expand_dims(best_anchors, 1)
    best_anchors = tf.tile(best_anchors, [1, 4, 1])
    word_poly = word_poly + best_anchors

    imh = tf.shape(image)[0]
    imw = tf.shape(image)[1]

    max_side = tf.maximum(imh, imw)
    pad_y = (max_side - imh) / 2
    pad_x = (max_side - imw) / 2
    square_scale = max_side / image_size

    char_poly *= square_scale
    word_poly *= square_scale

    diff = [pad_x, pad_y]
    char_poly -= diff
    word_poly -= diff

    boxes = tf.stack([y0, x0, y1, x1], axis=1)
    boxes = tf.expand_dims(boxes, 0)

    colors = tf.random.uniform([tf.shape(true_bboxes)[0], 4], minval=0, maxval=1, dtype=tf.dtypes.float32)

    image = tf.expand_dims(image, 0)
    image = tf.image.draw_bounding_boxes(image,  boxes, colors)
    image = tf.squeeze(image, 0)
    return filename, image

def non_max_suppression(coords, scores, max_ret, iou_threshold):
    ymin, xmin, ymax, xmax = tf.split(coords, num_or_size_splits=4, axis=1)
    ymax = tf.squeeze(ymax, 1)
    xmax = tf.squeeze(xmax, 1)
    ymin = tf.squeeze(ymin, 1)
    xmin = tf.squeeze(xmin, 1)

    area = (xmax - xmin) * (ymax - ymin)

    idxs = tf.argsort(scores, direction='ASCENDING', stable=False)

    max_idx = tf.minimum(tf.shape(idxs)[0], max_ret)

    pick = tf.TensorArray(tf.int32, size=max_idx)
    written = 0

    for idx in range(max_idx):
        last_idx = tf.shape(idxs)[0] - 1
        if idx >= last_idx:
            break

        #-----------------------------------------------------------------------
        # Grab the last index (ie. the most confident detection), remove it from
        # the list of indices to process, and put it on the list of picks
        #-----------------------------------------------------------------------
        i = idxs[last_idx]
        idxs = idxs[:last_idx]

        pick = pick.write(idx, i)
        written += 1
        suppress = []

        xmin_idx = tf.gather(xmin, idxs)
        xmax_idx = tf.gather(xmax, idxs)
        ymin_idx = tf.gather(ymin, idxs)
        ymax_idx = tf.gather(ymax, idxs)

        xmin_i = tf.gather(xmin, i)
        xmax_i = tf.gather(xmax, i)
        ymin_i = tf.gather(ymin, i)
        ymax_i = tf.gather(ymax, i)

        #-----------------------------------------------------------------------
        # Figure out the intersection with the remaining windows
        #-----------------------------------------------------------------------
        xxmin = tf.maximum(xmin_i, xmin_idx)
        xxmax = tf.minimum(xmax_i, xmax_idx)
        yymin = tf.maximum(ymin_i, ymin_idx)
        yymax = tf.minimum(ymax_i, ymax_idx)

        w = tf.maximum(0., xxmax-xxmin+1)
        h = tf.maximum(0., yymax-yymin+1)
        intersection = w*h

        #-----------------------------------------------------------------------
        # Compute IOU and suppress indices with IOU higher than a threshold
        #-----------------------------------------------------------------------
        area_i = tf.gather(area, i)
        area_idx = tf.gather(area, idxs)
        union = area_i + area_idx - intersection
        iou = intersection/union

        nonoverlap_index = tf.where(iou <= iou_threshold)
        nonoverlap_index = tf.squeeze(nonoverlap_index, 1)

        idxs = tf.gather(idxs, nonoverlap_index)

    pick = pick.stack()[:written]
    return pick
    #return tf.gather(coords, pick), tf.gather(scores, pick)

def polygon2bbox(poly):
    # polygon shape [N, 4, 2]

    x = poly[..., 0]
    y = poly[..., 1]

    xmin = tf.math.reduce_min(x, axis=1)
    ymin = tf.math.reduce_min(y, axis=1)
    xmax = tf.math.reduce_max(x, axis=1)
    ymax = tf.math.reduce_max(y, axis=1)

    bbox = tf.stack([ymin, xmin, ymax, xmax], 1)
    return bbox

def per_image_supression(pred_values, image_size, dictionary_size, all_anchors):
    char_boundary_start = 0
    word_boundary_start = dictionary_size + 1 + 4 * 2

    # true tensors
    pred_char = pred_values[..., char_boundary_start : char_boundary_start + word_boundary_start]
    pred_word = pred_values[..., word_boundary_start : ]

    pred_char_obj = pred_char[..., 0]
    pred_char_poly = pred_char[..., 1 : 9]
    pred_char_letters_dim = pred_char[..., 10 :]
    pred_char_letters = tf.argmax(pred_char_letters_dim, axis=-1)
    pred_char_letters_prob = tf.reduce_max(pred_char_letters_dim, axis=-1)

    pred_word_obj = pred_word[..., 0]
    pred_word_poly = pred_word[..., 1 : 9]

    char_index = tf.where(pred_char_obj > 0)
    word_index = tf.where(pred_word_obj > 0)

    char_poly = tf.gather(pred_char_poly, char_index)
    char_poly = tf.reshape(char_poly, [-1, 4, 2])
    word_poly = tf.gather(pred_word_poly, word_index).numpy()
    word_poly = tf.reshape(word_poly, [-1, 4, 2])

    best_anchors = tf.gather(all_anchors[..., :2], char_index)
    best_anchors = tf.tile(best_anchors, [1, 4, 1])
    char_poly = char_poly + best_anchors

    best_anchors = tf.gather(all_anchors[..., :2], word_index)
    best_anchors = tf.tile(best_anchors, [1, 4, 1])
    word_poly = word_poly + best_anchors

    char_obj = tf.gather(pred_char_obj, char_index)
    word_obj = tf.gather(pred_word_obj, word_index)

    char_letters = tf.gather(pred_char_letters, char_index)
    char_letters_prob = tf.gather(pred_char_letters_prob, char_index)

    # polygon [N, 4] -> [N, [ymin, xmin, ymax, xmax]]
    char_bboxes = polygon2bbox(char_poly)
    word_bboxes = polygon2bbox(word_poly)

    scores_to_sort = tf.squeeze(char_obj * char_letters_prob, 1)
    bboxes_to_sort = char_bboxes
    if True:
        selected_indexes = tf.image.non_max_suppression(bboxes_to_sort, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)
    else:
        selected_indexes = non_max_suppression(bboxes_to_sort, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)

    char_poly = tf.gather(char_poly, selected_indexes)
    char_obj = tf.gather(char_obj, selected_indexes)
    char_letters = tf.gather(char_letters, selected_indexes)
    char_letters_prob = tf.gather(char_letters_prob, selected_indexes)

    scores_to_sort = char_obj * char_letters_prob
    #scores_to_sort = char_obj
    scores_to_sort = tf.squeeze(scores_to_sort, 1)
    _, best_index = tf.math.top_k(scores_to_sort, tf.minimum(FLAGS.max_ret, tf.shape(scores_to_sort)[0]), sorted=True)

    char_poly = tf.gather(char_poly, best_index)
    char_obj = tf.gather(char_obj, best_index)
    char_obj = tf.squeeze(char_obj, 1)
    char_letters = tf.gather(char_letters, best_index)
    char_letters = tf.squeeze(char_letters, 1)

    to_add = tf.maximum(FLAGS.max_ret - tf.shape(char_obj)[0], 0)
    char_poly = tf.pad(char_poly, [[0, to_add], [0, 0], [0, 0]] , 'CONSTANT')
    char_obj = tf.pad(char_obj, [[0, to_add]], 'CONSTANT')
    char_letters = tf.pad(char_letters, [[0, to_add]], 'CONSTANT')

    return char_poly, char_obj, char_letters

def eval_step_logits(model, images, image_size, all_anchors, dictionary_size):
    pred_values = model(images, training=False)

    return tf.map_fn(lambda out: per_image_supression(out, image_size, dictionary_size, all_anchors),
                     pred_values,
                     parallel_iterations=FLAGS.num_cpus,
                     back_prop=False,
                     infer_shape=False,
                     dtype=(tf.float32, tf.float32, tf.int64))

def run_eval(model, dataset, image_size, dst_dir, all_anchors, dictionary_size, dictionary):
    num_files = 0
    dump_js = []
    for filenames, images in dataset:
        start_time = time.time()
        poly_batch, objs_batch, letters_batch = eval_step_logits(model, images, image_size, all_anchors, dictionary_size)
        num_files += len(filenames)
        time_per_image_ms = (time.time() - start_time) / len(filenames) * 1000

        logger.info('batch images: {}, total_processed: {}, time_per_image: {:.1f} ms'.format(len(filenames), num_files, time_per_image_ms))


        poly_batch = poly_batch.numpy()
        objs_batch = objs_batch.numpy()
        letters_batch = letters_batch.numpy()

        for filename, nn_image, polys, objs, letters in zip(filenames, images, poly_batch, objs_batch, letters_batch):
            filename = str(filename.numpy(), 'utf8')

            if os.path.isabs(filename):
                image = cv2.imread(filename)
            else:
                image_filename = '/shared2/object_detection/datasets/text/synth_text/SynthText/{}'.format(filename)
                image = cv2.imread(image_filename)

            base_filename = os.path.basename(filename)

            imh, imw = image.shape[:2]
            max_side = max(imh, imw)
            pad_y = (max_side - imh) / 2
            pad_x = (max_side - imw) / 2

            square_scale = max_side / image_size
            polys *= square_scale

            diff = [pad_x, pad_y]
            polys -= diff

            anns = []

            js = {
                'filename': filename,
            }
            js_anns = []

            fig = None
            text_ax = None
            image_ax = None
            if not FLAGS.do_not_save_images:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # only needed for images opened with imread()

                rows = 1
                columns = 2
                scale = 5
                fig = plt.figure(figsize=(columns*scale, rows*scale))

                image_ax = fig.add_subplot(rows, columns, 1)
                image_ax.get_xaxis().set_visible(False)
                image_ax.get_yaxis().set_visible(False)

                image_ax.set_autoscale_on(True)
                image_ax.imshow(image)

                text_ax = fig.add_subplot(rows, columns, 2)
                text_ax.set_autoscale_on(True)
                text_ax.get_xaxis().set_visible(False)
                text_ax.get_yaxis().set_visible(False)


            for crop_idx, (poly, obj, letter) in enumerate(zip(polys, objs, letters)):
                if obj < FLAGS.min_obj_score:
                    break

                if letter == 0:
                    letter = ' '
                else:
                    letter = dictionary[letter - 1]

                #logger.info('{}: poly: {}, obj: {:.3f}, letter: {}'.format(filename, poly, obj, letter))

                ann_js = {
                    'poly': poly.tolist(),
                    'letter': letter,
                    'objectness': float(obj),
                }

                anns.append((None, poly, None))
                js_anns.append(ann_js)

                if not FLAGS.do_not_save_images:
                    xmin = poly[0][0]
                    ymin = poly[0][1]

                    x = xmin/image.shape[1]
                    y = 1 - ymin/image.shape[0]
                    text = text_ax.text(x, y, letter, verticalalignment='top', color='black', fontsize=8, weight='normal')

                    poly = poly.reshape([-1, 2])
                    for xy in poly:
                        cr = patches.Circle(xy, 2, color='r')
                        image_ax.add_artist(cr)

                    poly = patches.Polygon(poly, fill=False)
                    image_ax.add_artist(poly)


            js['annotations'] = js_anns
            dump_js.append(js)

            if not FLAGS.do_not_save_images:
                dst_filename = os.path.join(FLAGS.output_dir, base_filename)

                plt.axis('off')
                plt.savefig(dst_filename)
                plt.close(fig)

    json_fn = os.path.join(FLAGS.output_dir, 'results.json')
    logger.info('Saving {} objects into {}'.format(len(dump_js), json_fn))

    with open(json_fn, 'w') as fout:
        json.dump(dump_js, fout)


def run_inference():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    if FLAGS.save_crops_dir:
        os.makedirs(FLAGS.save_crops_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, 'infdet.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    num_images = len(FLAGS.filenames)

    dtype = tf.float32
    image_size = FLAGS.image_size
    dictionary_size = len(FLAGS.dictionary) + 1
    model = encoder.create_model(FLAGS.model_name, dictionary_size)
    if model.output_sizes is None:
        dummy_input = tf.ones((int(FLAGS.batch_size), image_size, image_size, 3), dtype=dtype)
        model(dummy_input, training=False)
        logger.info('image_size: {}, model output sizes: {}'.format(image_size, model.output_sizes))

    all_anchors, all_grid_xy, all_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

    checkpoint = tf.train.Checkpoint(model=model)

    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    status = checkpoint.restore(checkpoint_prefix)
    if FLAGS.skip_checkpoint_assertion:
        status.expect_partial()
    else:
        status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    if FLAGS.dataset_type == 'files':
        ds = tf.data.Dataset.from_tensor_slices((FLAGS.filenames))
        ds = ds.map(lambda fn: tf_read_image(fn, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
    elif FLAGS.dataset_type == 'filelist':
        filenames = []
        for fn in FLAGS.filenames:
            with open(fn, 'r') as fin:
                for line in fin:
                    if line[-1] == '\n':
                        line = line[:-1]
                    filenames.append(line)
        ds = tf.data.Dataset.from_tensor_slices((filenames))
        ds = ds.map(lambda fn: tf_read_image(fn, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
    elif FLAGS.dataset_type == 'tfrecords':
        from detection import unpack_tfrecord

        filenames = []
        for fn in os.listdir(FLAGS.eval_tfrecord_dir):
            fn = os.path.join(FLAGS.eval_tfrecord_dir, fn)
            if os.path.isfile(fn):
                filenames.append(fn)

        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
        ds = ds.map(lambda record: unpack_tfrecord(record, all_anchors,
                    image_size, False,
                    FLAGS.data_format),
            num_parallel_calls=16)
        ds = ds.map(lambda filename, image, true_values: tf_left_needed_dimensions_from_tfrecord(image_size, all_anchors, filename, image, true_values, dictionary_size),
            num_parallel_calls=16)

    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    logger.info('Dataset has been created: num_images: {}, model_name: {}'.format(num_images, FLAGS.model_name))

    run_eval(model, ds, image_size, FLAGS.output_dir, all_anchors, dictionary_size, FLAGS.dictionary)

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    if not FLAGS.checkpoint and not FLAGS.checkpoint_dir:
        logger.critical('You must provide either checkpoint or checkpoint dir')
        exit(-1)

    try:
        run_inference()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
