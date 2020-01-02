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

from shapely.geometry import Polygon

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
parser.add_argument('--save_crops_dir', type=str, help='Directory to save slightly upscaled text crops')
parser.add_argument('--dataset_type', type=str, choices=['files', 'mscoco_tfrecords', 'filelist'], default='files', help='Dataset type')
parser.add_argument('--image_size', type=int, required=True, help='Use this image size, if 0 - use default')
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')

default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
parser.add_argument('--dictionary', type=str, default=default_char_dictionary, help='Dictionary to use')
FLAGS = parser.parse_args()

def scale_norm_image(image, image_size, bboxes, dtype):
    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2, tf.int32),
                tf.cast((mx - orig_image_width) / 2, tf.int32),
                mx_int,
                mx_int)

    if FLAGS.dataset_type == 'mscoco_dataset':
        cx, cy, h, w = tf.split(bboxes, num_or_size_splits=4, axis=1)
        diff_x = (mx - orig_image_width) / 2
        diff_y = (mx - orig_image_height) / 2
        diff = tf.convert_to_tensor([diff_x, diff_y, 0, 0])

        bboxes += diff

        square_scale = image_size / mx
        bboxes *= square_scale

    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, dtype)

    image -= 128
    image /= 128

    return image, bboxes

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image, _ = scale_norm_image(image, image_size, 0, dtype)
    return filename, image, 0, 0

def mscoco_unpack_tfrecord(record, anchors_all, image_size, is_training, dict_table, dictionary_size, data_format, dtype):
    features = tf.io.parse_single_example(record,
            features={
                'image_id': tf.io.FixedLenFeature([], tf.int64),
                'filename': tf.io.FixedLenFeature([], tf.string),
                'true_labels': tf.io.FixedLenFeature([], tf.string),
                'true_bboxes': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            })

    filename = features['filename']

    text_labels = tf.strings.split(features['true_labels'], '<SEP>')

    orig_bboxes = tf.io.decode_raw(features['true_bboxes'], tf.float32)
    orig_bboxes = tf.reshape(orig_bboxes, [-1, 4])

    cx, cy, h, w = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)

    image = tf.image.decode_jpeg(features['image'], channels=3)
    image, bboxes = scale_norm_image(image, image_size, bboxes, dtype)

    ymin, xmin, ymax, xmax = tf.split(bboxes, num_or_size_splits=4, axis=1)
    w = xmax - xmin
    h = ymax - ymin

    p0 = tf.concat([xmin, ymin], axis=1)
    p1 = tf.concat([xmin + w, ymin], axis=1)
    p2 = tf.concat([xmin + w, ymin + h], axis=1)
    p3 = tf.concat([xmin, ymin + h], axis=1)

    word_poly = tf.stack([p0, p1, p2, p3], axis=1)

    to_add = tf.maximum(500 - tf.shape(word_poly)[0], 0)
    word_poly = tf.pad(word_poly, [[0, to_add], [0, 0], [0, 0]] , 'CONSTANT')
    text_labels = tf.pad(text_labels, [[0, to_add]], 'CONSTANT')


    return filename, image, word_poly, text_labels

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

def nms_for_poly(poly, obj, scores_to_sort):
    # polygon [N, 4] -> [N, [ymin, xmin, ymax, xmax]]
    bboxes_to_sort = anchors_gen.polygon2bbox(poly, want_yx=True)

    if True:
        selected_indexes = tf.image.non_max_suppression(bboxes_to_sort, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)
    else:
        selected_indexes = non_max_suppression(bboxes_to_sort, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)

    poly = tf.gather(poly, selected_indexes)
    obj = tf.gather(obj, selected_indexes)
    return poly, obj, selected_indexes

def sort_and_pad_for_poly(poly, obj, scores_to_sort):
    _, best_index = tf.math.top_k(scores_to_sort, tf.minimum(FLAGS.max_ret, tf.shape(scores_to_sort)[0]), sorted=True)

    poly = tf.gather(poly, best_index)
    obj = tf.gather(obj, best_index)
    obj = tf.squeeze(obj, 1)

    to_add = tf.maximum(FLAGS.max_ret - tf.shape(obj)[0], 0)
    poly = tf.pad(poly, [[0, to_add], [0, 0], [0, 0]] , 'CONSTANT')
    obj = tf.pad(obj, [[0, to_add]], 'CONSTANT')
    return poly, obj, best_index

def per_image_supression(pred_values, image_size, dictionary_size, all_anchors):
    char_boundary_start = 0
    word_boundary_start = dictionary_size + 1 + 4 * 2

    # true tensors
    pred_char = pred_values[..., char_boundary_start : char_boundary_start + word_boundary_start]
    pred_word = pred_values[..., word_boundary_start : ]

    pred_char_obj = tf.math.sigmoid(pred_char[..., 0])
    pred_char_poly = pred_char[..., 1 : 9]
    pred_char_letters_dim = tf.nn.softmax(pred_char[..., 10 :], axis=-1)
    pred_char_letters = tf.argmax(pred_char_letters_dim, axis=-1)
    pred_char_letters_prob = tf.reduce_max(pred_char_letters_dim, axis=-1)

    pred_word_obj = tf.math.sigmoid(pred_word[..., 0])
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

    char_poly, char_obj, char_index = nms_for_poly(char_poly, char_obj, tf.squeeze(char_obj * char_letters_prob, 1))
    char_letters = tf.gather(char_letters, char_index)
    char_letters_prob = tf.gather(char_letters_prob, char_index)
    char_poly, char_obj, char_index = sort_and_pad_for_poly(char_poly, char_obj, tf.squeeze(char_obj * char_letters_prob, 1))
    char_letters = tf.gather(char_letters, char_index)
    char_letters = tf.squeeze(char_letters, 1)
    char_letters_prob = tf.gather(char_letters_prob, char_index)

    to_add = tf.maximum(FLAGS.max_ret - tf.shape(char_index)[0], 0)
    char_letters = tf.pad(char_letters, [[0, to_add]], 'CONSTANT')


    word_poly, word_obj, word_index = nms_for_poly(word_poly, word_obj, tf.squeeze(word_obj, 1))
    word_poly, word_obj, word_index = sort_and_pad_for_poly(word_poly, word_obj, tf.squeeze(word_obj, 1))

    return char_obj, char_poly, char_letters, word_obj, word_poly

def eval_step_logits(model, images, image_size, all_anchors, dictionary_size):
    pred_values = model(images, training=False)

    return tf.map_fn(lambda out: per_image_supression(out, image_size, dictionary_size, all_anchors),
                     pred_values,
                     parallel_iterations=FLAGS.num_cpus,
                     back_prop=False,
                     infer_shape=False,
                     dtype=(tf.float32, tf.float32, tf.int64, tf.float32, tf.float32))

def draw_poly(image_ax, objs, polys, color='black'):
    js_anns = []

    for obj, poly in zip(objs, polys):
        if obj < FLAGS.min_obj_score:
            break

        ann_js = {
            'poly': poly.tolist(),
            'objectness': float(obj),
        }
        js_anns.append(ann_js)

        if not FLAGS.do_not_save_images:
            poly = poly.reshape([-1, 2])
            for xy in poly:
                cr = patches.Circle(xy, 2, color='r')
                image_ax.add_artist(cr)

            poly = patches.Polygon(poly, fill=False, color=color)
            image_ax.add_artist(poly)

    return js_anns

def draw_char_letters(text_ax, char_objs, char_polys, char_letters, dictionary, image_shape):
    js_anns = []

    for obj, poly, letter in zip(char_objs, char_polys, char_letters):
        if obj < FLAGS.min_obj_score:
            break

        if letter == 0:
            letter = ' '
        else:
            letter = dictionary[letter - 1]

        ann_js = {
            'poly': poly.tolist(),
            'letter': letter,
            'objectness': float(obj),
        }
        js_anns.append(ann_js)

        if not FLAGS.do_not_save_images:
            xmin = poly[0][0]
            ymin = poly[0][1]

            x = xmin/image_shape[1]
            y = 1 - ymin/image_shape[0]
            text = text_ax.text(x, y, letter, verticalalignment='top', color='black', fontsize=8, weight='normal')

    return js_anns

def run_eval(model, dataset, image_size, dst_dir, all_anchors, dictionary_size, dictionary):
    num_files = 0
    dump_js = []
    for filenames, images, true_word_poly_batch, true_text_labels_batch in dataset:
        start_time = time.time()
        char_obj_batch, char_poly_batch, char_letters_batch, word_obj_batch, word_poly_batch = eval_step_logits(model, images, image_size, all_anchors, dictionary_size)
        num_files += len(filenames)
        time_per_image_ms = (time.time() - start_time) / len(filenames) * 1000

        logger.info('batch images: {}, total_processed: {}, time_per_image: {:.1f} ms'.format(len(filenames), num_files, time_per_image_ms))


        char_obj_batch = char_obj_batch.numpy()
        char_poly_batch = char_poly_batch.numpy()
        char_letters_batch = char_letters_batch.numpy()

        word_obj_batch = word_obj_batch.numpy()
        word_poly_batch = word_poly_batch.numpy()

        for filename, nn_image, char_objs, char_polys, char_letters, word_objs, word_polys, true_word_poly, true_text_labels in zip(filenames, images,
                                                                                                                                    char_obj_batch, char_poly_batch, char_letters_batch,
                                                                                                                                    word_obj_batch, word_poly_batch,
                                                                                                                                    true_word_poly_batch, true_text_labels_batch):
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
            char_polys *= square_scale
            word_polys *= square_scale

            diff = [pad_x, pad_y]
            char_polys -= diff
            word_polys -= diff

            if FLAGS.dataset_type == 'mscoco_tfrecords':
                if False:
                    ious = anchors_gen.box_iou(char_polys, true_word_poly)
                    # [char, word] -> [word, char]
                    ious = tf.transpose(ious, [1, 0])

                    word_char_intersection_threshold = 0.1
                    iou_intersections = tf.where(ious > word_char_intersection_threshold, 1, 0)
                    word_intersections = tf.reduce_sum(iou_intersections, 1)

                    word_intersections = word_intersections.numpy()

                new_char_polys = []
                new_word_polys = []

                cps = [Polygon(cp) for cp in char_polys]
                true_word_poly = true_word_poly.numpy()
                true_text_labels = true_text_labels.numpy()

                for wp, text_label in zip(true_word_poly, true_text_labels):
                    if len(wp) == 0:
                        break

                    text_label = str(text_label, 'utf8')
                    if len(text_label) == 0 or text_label == '<SKIP>':
                        continue

                    wp = Polygon(wp)

                    good_cp = []
                    for char_obj, cp in zip(char_objs, cps):
                        if char_obj < FLAGS.min_obj_score:
                            continue

                        if (wp.intersects(cp)):
                            good_cp.append(cp)

                            if len(good_cp) > len(text_label):
                                break

                    if len(good_cp) == len(text_label):
                        good_cp = [list(cp.exterior.coords)[:4] for cp in good_cp]
                        new_cp = np.concatenate(good_cp, 0)
                        new_char_polys.append(good_cp)

                        wp = list(wp.exterior.coords)[:4]
                        logger.info('{}: text_label: {}, good_cp: {}'.format(filename, text_label, len(good_cp)))

                        wp = np.array(wp)

                        new_word_polys.append(wp)

                if len(new_char_polys):
                    char_polys = np.concatenate(new_char_polys, 0)
                    word_polys = np.stack(new_word_polys, 0)
                    logger.info('{}: char_poly: {}, word_poly: {}'.format(filename, char_polys.shape, word_polys.shape))
                    #word_polys = true_word_poly
                else:
                    continue

            js = {
                'filename': filename,
            }

            fig = None
            text_ax = None
            image_ax = None
            if not FLAGS.do_not_save_images:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # only needed for images opened with imread()

                rows = 1
                columns = 2
                scale = 10
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


            char_js_anns = draw_char_letters(text_ax, char_objs, char_polys, char_letters, dictionary, image.shape)

            _ = draw_poly(image_ax, char_objs, char_polys, color='orange')
            word_js_anns = draw_poly(image_ax, word_objs, word_polys, color='green')

            js['char'] = char_js_anns
            js['word'] = word_js_anns
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

    dictionary_size, dict_table = anchors_gen.create_lookup_table(FLAGS.dictionary)

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
    elif FLAGS.dataset_type == 'mscoco_tfrecords':
        filenames = []
        for dirname in FLAGS.filenames:
            for fn in os.listdir(dirname):
                fn = os.path.join(dirname, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
        ds = ds.map(lambda record: mscoco_unpack_tfrecord(record, all_anchors, image_size, False, dict_table, dictionary_size, FLAGS.data_format, dtype),
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
