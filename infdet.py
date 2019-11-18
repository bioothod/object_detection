import argparse
import cv2
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

from collections import defaultdict

import anchors_gen
import coco
import encoder
import image as image_draw
import loss
import preprocess_ssd

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--eval_coco_annotations', type=str, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--eval_coco_data_dir', type=str, default='/', help='Path to MS COCO dataset: image directory')
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
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')
FLAGS = parser.parse_args()

def normalize_image(image, dtype):
    image = tf.cast(image, dtype)

    image = preprocess_ssd.normalize_image(image)
    return image

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_image_height = tf.cast(tf.shape(image)[0], dtype)
    orig_image_width = tf.cast(tf.shape(image)[1], dtype)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image, tf.cast((mx - orig_image_height) / 2, tf.int32), tf.cast((mx - orig_image_width) / 2, tf.int32), mx_int, mx_int)

    image = tf.image.resize(image, [image_size, image_size])

    image = normalize_image(image, dtype)

    return filename, image

def tf_left_needed_dimensions_from_tfrecord(image_size, anchors_all, output_xy_grids, output_ratios, filename, image_id, image, true_values):
    non_background_index = tf.where(tf.not_equal(true_values[..., 4], 0))

    sampled_true_values = tf.gather_nd(true_values, non_background_index)

    true_obj = sampled_true_values[:, 5]
    true_bboxes = sampled_true_values[:, 0:4]
    true_labels = sampled_true_values[:, 5:]
    true_labels = tf.argmax(true_labels, axis=1)

    grid_xy = tf.gather_nd(output_xy_grids, non_background_index)
    ratios = tf.gather_nd(output_ratios, non_background_index)
    ratios = tf.expand_dims(ratios, -1)
    anchors_wh = tf.gather_nd(anchors_all, non_background_index)

    true_xy = (true_bboxes[..., 0:2] + grid_xy) * ratios
    true_wh = tf.math.exp(true_bboxes[..., 2:4]) * anchors_wh[..., 2:4]

    cx = true_xy[..., 0]
    cy = true_xy[..., 1]
    w = true_wh[..., 0]
    h = true_wh[..., 1]

    x0 = cx - w/2
    x1 = cx + w/2
    y0 = cy - h/2
    y1 = cy + h/2

    x0 /= image_size
    y0 /= image_size
    x1 /= image_size
    y1 /= image_size

    boxes = tf.stack([y0, x0, y1, x1], axis=1)
    boxes = tf.expand_dims(boxes, 0)

    colors = tf.random.uniform([tf.shape(true_bboxes)[0], 4], minval=0, maxval=1, dtype=tf.dtypes.float32)

    image = tf.expand_dims(image, 0)
    image = tf.image.draw_bounding_boxes(image,  boxes, colors)
    image = tf.squeeze(image, 0)
    return filename, image

def tf_left_needed_dimensions(image_size, filename, image_id, image, true_bboxes, true_labels):
    pos_indexes = tf.where(true_labels > 0)
    pos_bboxes = tf.gather_nd(true_bboxes, pos_indexes)

    cx = pos_bboxes[:, 0]
    cy = pos_bboxes[:, 1]
    h = pos_bboxes[:, 2]
    w = pos_bboxes[:, 3]

    x0 = cx - w/2
    x1 = cx + w/2
    y0 = cy - h/2
    y1 = cy + h/2

    x0 /= image_size
    y0 /= image_size
    x1 /= image_size
    y1 /= image_size

    boxes = tf.stack([y0, x0, y1, x1], axis=1)
    boxes = tf.expand_dims(boxes, 0)

    image = tf.expand_dims(image, 0)

    colors = tf.random.uniform([tf.shape(pos_bboxes)[0], 4], minval=0, maxval=1, dtype=tf.dtypes.float32)

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

def per_image_supression(logits, image_size):
    coords, objectness = logits

    non_background_index = tf.where(tf.greater(objectness, FLAGS.min_obj_score))

    non_background_index = tf.squeeze(non_background_index, 1)

    selected_coords = tf.gather(coords, non_background_index)
    selected_objs = tf.gather(objectness, non_background_index)

    ret_coords, ret_cat_ids, ret_objs = [], [], []

    cx, cy, w, h = tf.split(selected_coords, num_or_size_splits=4, axis=1)
    cx = tf.squeeze(cx, 1)
    cy = tf.squeeze(cy, 1)
    w = tf.squeeze(w, 1)
    h = tf.squeeze(h, 1)

    small_cond = tf.logical_and(h >= FLAGS.min_size, w >= FLAGS.min_size)
    large_cond = tf.logical_and(h < image_size*2, w < image_size*2)
    index = tf.where(tf.logical_and(small_cond, large_cond))
    selected_coords = tf.gather(selected_coords, index)
    selected_coords = tf.squeeze(selected_coords, 1)
    selected_objs = tf.gather(selected_objs, index)
    selected_objs = tf.squeeze(selected_objs, 1)

    cx, cy, w, h = tf.split(selected_coords, num_or_size_splits=4, axis=1)
    cx = tf.squeeze(cx, 1)
    cy = tf.squeeze(cy, 1)
    w = tf.squeeze(w, 1)
    h = tf.squeeze(h, 1)

    #tf.print('cx:', cx, 'cy:', cy, 'w:', w, 'h:', h)

    xmin = cx - w/2
    xmax = cx + w/2
    ymin = cy - h/2
    ymax = cy + h/2

    xmin = tf.maximum(0., xmin)
    ymin = tf.maximum(0., ymin)
    xmax = tf.minimum(float(image_size), xmax)
    ymax = tf.minimum(float(image_size), ymax)

    coords_yx = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    scores_to_sort = selected_objs
    if True:
        selected_indexes = tf.image.non_max_suppression(coords_yx, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)
    else:
        selected_indexes = non_max_suppression(coords_yx, scores_to_sort, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)

    selected_coords = tf.gather(coords_yx, selected_indexes)
    selected_objs = tf.gather(selected_objs, selected_indexes)

    scores_to_sort = selected_objs
    _, best_index = tf.math.top_k(scores_to_sort, tf.minimum(FLAGS.max_ret, tf.shape(scores_to_sort)[0]), sorted=True)

    best_coords = tf.gather(selected_coords, best_index)
    best_objs = tf.gather(selected_objs, best_index)

    to_add = tf.maximum(FLAGS.max_ret - tf.shape(best_objs)[0], 0)
    best_coords = tf.pad(best_coords, [[0, to_add], [0, 0]] , 'CONSTANT')
    best_objs = tf.pad(best_objs, [[0, to_add]], 'CONSTANT')

    return best_coords, best_objs

@tf.function
def eval_step_logits(model, images, image_size, all_anchors, all_grid_xy, all_ratios):
    pred_values = model(images, training=False)

    pred_objs = tf.math.sigmoid(pred_values[..., 4])

    pred_bboxes = pred_values[..., 0:4]

    all_ratios = tf.expand_dims(all_ratios, -1)

    pred_xy = (tf.sigmoid(pred_bboxes[..., 0:2]) + all_grid_xy) * all_ratios
    pred_wh = tf.math.exp(pred_bboxes[..., 2:4]) * all_anchors[..., 2:4]

    pred_bboxes = tf.concat([pred_xy, pred_wh], axis=-1)

    return tf.map_fn(lambda out: per_image_supression(out, image_size),
                     (pred_bboxes, pred_objs),
                     parallel_iterations=FLAGS.num_cpus,
                     back_prop=False,
                     infer_shape=False,
                     dtype=(tf.float32, tf.float32))

def run_eval(model, dataset, num_images, image_size, dst_dir, all_anchors, all_grid_xy, all_ratios):
    num_files = 0
    dump_js = []
    for filenames, images in dataset:
        start_time = time.time()
        coords_batch, objs_batch = eval_step_logits(model, images, image_size, all_anchors, all_grid_xy, all_ratios)
        num_files += len(filenames)
        time_per_image_ms = (time.time() - start_time) / len(filenames) * 1000

        logger.info('batch images: {}, total_processed: {}, time_per_image: {:.1f} ms'.format(len(filenames), num_files, time_per_image_ms))

        for filename, nn_image, coords, objectness in zip(filenames, images, coords_batch, objs_batch):
            filename = str(filename.numpy(), 'utf8')
            image = cv2.imread(filename)

            base_filename = os.path.basename(filename)

            imh, imw = image.shape[:2]
            max_side = max(imh, imw)
            pad_y = (max_side - imh) / 2
            pad_x = (max_side - imw) / 2
            square_scale = max_side / image_size

            anns = []

            js = {
                'filename': filename,
            }
            js_anns = []

            for crop_idx, (coord, objs) in enumerate(zip(coords, objectness)):
                objs = float(objs.numpy())
                if objs < FLAGS.min_obj_score:
                    break

                bb = coord.numpy().astype(float)

                scaled_bb = bb * square_scale

                ymin, xmin, ymax, xmax = scaled_bb
                xmin -= pad_x
                xmax -= pad_x
                ymin -= pad_y
                ymax -= pad_y

                #ymin, xmin, ymax, xmax = bb
                bb = [xmin, ymin, xmax, ymax]


                logger.info('{}: bbox: {}, obj: {:.3f}'.format(filename, bb, objs))

                ann_js = {
                    'bbox': bb,
                    'objectness': objs,
                }

                js_anns.append(ann_js)
                anns.append((bb, None, '.'))

                if FLAGS.save_crops_dir:
                    output_filename = '{}_{}_{:.3f}.jpg'.format(base_filename, crop_idx, objs)
                    output_filename = os.path.join(FLAGS.save_crops_dir, output_filename)

                    scaled_crop_bb = [int(d * 1.1) for d in bb]
                    xmin, ymin, xmax, ymax = scaled_crop_bb
                    crop = image[ymin:ymax+1, xmin:xmax+1, :]

                    cv2.imwrite(output_filename, crop)


            js['annotations'] = js_anns
            dump_js.append(js)

            if not FLAGS.do_not_save_images:
                #image = preprocess_ssd.denormalize_image(nn_image).numpy().astype(np.uint8)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # only needed for original images
                dst_filename = os.path.join(FLAGS.output_dir, base_filename)
                image_draw.draw_im(image, anns, dst_filename, {})


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
    model = encoder.create_model(FLAGS.model_name)
    image_size = model.image_size
    all_anchors, all_grid_xy, all_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

    checkpoint = tf.train.Checkpoint(model=model)

    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if FLAGS.freeze:
        with tf.compat.v1.Session() as sess:
            images = tf.keras.layers.Input(shape=(image_size * image_size * 3), name='input/images_rgb', dtype=tf.uint8)
            images = tf.reshape(images, [-1, image_size, image_size, 3])
            images = normalize_image(images, dtype)

            model = encoder.create_model(FLAGS.model_name)
            image_size = model.image_size
            all_anchors, all_grid_xy, all_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

            checkpoint = tf.train.Checkpoint(model=model)

            coords_batch, objs_batch = eval_step_logits(model, images, image_size, all_anchors, all_grid_xy, all_ratios)

            coords_batch = tf.identity(coords_batch, name='output/coords')
            objs_batch = tf.identity(objs_batch, name='output/objectness')

            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched().expect_partial()
            logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['output/coords', 'output/objectness'])

        output = '{}-{}.frozen.pb'.format(checkpoint_prefix, tf.__version__)
        filename = tf.io.write_graph(output_graph_def, os.path.dirname(output), os.path.basename(output), as_text=False)

        print('Saved graph as {}'.format(os.path.abspath(filename)))
        return


    status = checkpoint.restore(checkpoint_prefix)
    if FLAGS.skip_checkpoint_assertion:
        status.expect_partial()
    else:
        status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    if FLAGS.eval_coco_annotations:
        eval_base = coco.create_coco_iterable(FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, logger)

        num_images = len(eval_base)

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
        ds = ds.map(lambda record: unpack_tfrecord(record, all_anchors, all_grid_xy, all_ratios,
                    image_size, False,
                    FLAGS.data_format),
            num_parallel_calls=16)
        ds = ds.map(lambda filename, image_id, image, true_values, true_texts: tf_left_needed_dimensions_from_tfrecord(image_size, all_anchors, all_grid_xy, all_ratios,
                    filename, image_id, image, true_values),
            num_parallel_calls=16)

    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    logger.info('Dataset has been created: num_images: {}, model_name: {}'.format(num_images, FLAGS.model_name))

    run_eval(model, ds, num_images, image_size, FLAGS.output_dir, all_anchors, all_grid_xy, all_ratios)

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
