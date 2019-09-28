import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from collections import defaultdict

import anchors_gen
import coco
import image as image_draw
import loss
import map_iter
import preprocess
import yolo

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--eval_coco_annotations', type=str, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--eval_coco_data_dir', type=str, help='Path to MS COCO dataset: image directory')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--num_classes', type=int, required=True, help='Number of the output classes in the model')
parser.add_argument('--max_ret', type=int, default=100, help='Maximum number of returned boxes')
parser.add_argument('--min_score', type=float, default=0.7, help='Minimal class probability')
parser.add_argument('--min_size', type=float, default=10, help='Minimal size of the bounding box')
parser.add_argument('--iou_threshold', type=float, default=0.45, help='Minimal IoU threshold for non-maximum suppression')
parser.add_argument('--output_dir', type=str, required=True, help='Path to directory, where images will be stored')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--checkpoint_dir', type=str, help='Load model weights from the latest checkpoint in this directory')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--dataset_type', type=str, choices=['files', 'tfrecords'], default='files', help='Dataset type')
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')
FLAGS = parser.parse_args()

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_image_height = tf.cast(tf.shape(image)[0], dtype)
    orig_image_width = tf.cast(tf.shape(image)[1], dtype)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image, tf.cast((mx - orig_image_height) / 2, tf.int32), tf.cast((mx - orig_image_width) / 2, tf.int32), mx_int, mx_int)

    image = tf.cast(image, dtype)
    image -= 128.
    image /= 128.

    image = tf.image.resize_with_pad(image, image_size, image_size)

    return filename, image

def tf_left_needed_dimensions_from_tfrecord(image_size, num_classes, np_anchor_boxes, filename, image_id, image, true_values):
    scaled_size = image_size / anchors_gen.DOWNSAMPLE_RATIO
    output_splits = []
    offset = 0
    num_scales = 3
    num_boxes = 3
    for base_scale in range(num_scales):
        output_size = scaled_size * np.math.pow(2, base_scale)
        output_splits.append(int(output_size * output_size))

    anchors_reshaped = tf.reshape(np_anchor_boxes, [3, -1])

    image = tf.expand_dims(image, 0)

    true_values = tf.expand_dims(true_values, 0)

    true_values_list = list(tf.split(true_values, output_splits, axis=1))
    for output_idx, true_values in enumerate(true_values_list):
        output_size = int(scaled_size) * tf.math.pow(2, output_idx)

        true_values = tf.reshape(true_values, [-1, output_size, output_size, num_boxes, 4 + 1 + num_classes])

        non_background_index = tf.where(tf.not_equal(true_values[..., 4], 0))
        box_index = non_background_index[:, 3]

        sampled_true_values = tf.gather_nd(true_values, non_background_index)

        true_obj = sampled_true_values[:, 5]
        true_bboxes = sampled_true_values[:, 0:4]
        true_labels = sampled_true_values[:, 5:]
        true_labels = tf.argmax(true_labels, axis=1)

        anchors_wh = anchors_reshaped[output_idx, :]

        anchor_grid = loss._create_mesh_anchor(anchors_wh, tf.shape(true_bboxes)[0], output_size, output_size, 3)
        anchor_grid = tf.gather_nd(anchor_grid, non_background_index)

        true_xy = true_bboxes[..., 0:2] / tf.cast(output_size, tf.float32) * float(image_size)
        true_xy = tf.reshape(true_xy, [-1, 2])

        true_wh = tf.math.exp(true_bboxes[..., 2:4]) * anchor_grid
        true_wh = tf.reshape(true_wh, [-1, 2])

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
    ymax, xmax, ymin, xmin = tf.split(coords, num_or_size_splits=4, axis=1)
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

def per_image_supression(logits, num_classes):
    coords, scores, labels, objectness = logits

    non_background_index = tf.where(tf.logical_and(
                                        tf.greater(objectness, 0.5),
                                        tf.greater(scores, FLAGS.min_score)))
    tf.print('max_obj:', tf.reduce_max(objectness), 'max_score:', tf.reduce_max(scores, -1), 'scores:', tf.shape(scores))
    #non_background_index = tf.where(tf.greater(scores, FLAGS.min_score))
    non_background_index = tf.squeeze(non_background_index, 1)
    tf.print('non_background_index:', tf.shape(non_background_index), ':', non_background_index)
    tf.print('coords:', tf.shape(coords), 'scores:', tf.shape(scores), 'labels:', tf.shape(labels))
    sampled_coords = tf.gather(coords, non_background_index)
    sampled_scores = tf.gather(scores, non_background_index)
    sampled_labels = tf.gather(labels, non_background_index)

    tf.print('labels:', tf.shape(labels), 'scores:', tf.shape(scores), 'coords:', tf.shape(coords))
    tf.print('sampled_labels:', tf.shape(sampled_labels), 'sampled_scores:', tf.shape(sampled_scores), 'sampled_coords:', tf.shape(sampled_coords))
    tf.print('sampled_labels:', sampled_labels, 'sampled_scores:', sampled_scores, 'sampled_coords:', sampled_coords)

    ret_coords, ret_scores, ret_cat_ids = [], [], []
    for cat_id in range(0, num_classes):
        class_index = tf.where(tf.equal(sampled_labels, cat_id))
        #tf.print('class_index:', class_index)

        selected_scores = tf.gather_nd(sampled_scores, class_index)
        #logger.info('class_index: {}, sampled_labels: {}, sampled_scores: {}, selected_scores: {}'.format(class_index.shape, sampled_labels.shape, sampled_scores.shape, selected_scores.shape))

        selected_coords = tf.gather_nd(sampled_coords, class_index)

        #tf.print('selected_scores:', selected_scores, 'selected_coords:', selected_coords)

        cx, cy, w, h = tf.split(selected_coords, num_or_size_splits=4, axis=1)
        cx = tf.squeeze(cx, 1)
        cy = tf.squeeze(cy, 1)
        w = tf.squeeze(w, 1)
        h = tf.squeeze(h, 1)

        index = tf.where(tf.logical_and(h >= FLAGS.min_size, w >= FLAGS.min_size))
        selected_scores = tf.gather(selected_scores, index)
        selected_scores = tf.squeeze(selected_scores, 1)
        selected_coords = tf.gather(selected_coords, index)
        selected_coords = tf.squeeze(selected_coords, 1)

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

        coords_yx = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        if True:
            selected_indexes = tf.image.non_max_suppression(coords_yx, selected_scores, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)
        else:
            selected_indexes = non_max_suppression(coords_yx, selected_scores, FLAGS.max_ret, iou_threshold=FLAGS.iou_threshold)

        #logger.info('selected_indexes: {}, selected_coords: {}, selected_scores: {}'.format(selected_indexes, selected_coords, selected_scores))
        #selected_coords = tf.gather(selected_coords, selected_indexes)
        #selected_scores = tf.gather(selected_scores, selected_indexes)

        ret_coords.append(selected_coords)
        ret_scores.append(selected_scores)

        num = tf.shape(selected_scores)[0]
        tile = tf.tile([cat_id], [num])
        ret_cat_ids.append(tile)

    ret_coords = tf.concat(ret_coords, 0)
    ret_scores = tf.concat(ret_scores, 0)
    ret_cat_ids = tf.concat(ret_cat_ids, 0)

    #logger.info('ret_coords: {}, ret_scores: {}, ret_cat_ids: {}'.format(ret_coords, ret_scores, ret_cat_ids))

    best_scores, best_index = tf.math.top_k(ret_scores, tf.minimum(FLAGS.max_ret, tf.shape(ret_scores)[0]), sorted=True)

    best_coords = tf.gather(ret_coords, best_index)
    best_cat_ids = tf.gather(ret_cat_ids, best_index)

    #logger.info('best_coords: {}, best_scores: {}, best_cat_ids: {}'.format(best_coords, best_scores, best_cat_ids))

    to_add = tf.maximum(FLAGS.max_ret - tf.shape(best_scores)[0], 0)
    best_coords = tf.pad(best_coords, [[0, to_add], [0, 0]] , 'CONSTANT')
    best_scores = tf.pad(best_scores, [[0, to_add]], 'CONSTANT')
    best_cat_ids = tf.pad(best_cat_ids, [[0, to_add]], 'CONSTANT')


    #logger.info('ret_coords: {}, ret_scores: {}, ret_cat_ids: {}, best_index: {}, best_scores: {}, best_coords: {}, best_cat_ids: {}'.format(
    #    ret_coords, ret_scores, ret_cat_ids, best_index, best_scores, best_coords, best_cat_ids))

    return best_coords, best_scores, best_cat_ids

@tf.function
def eval_step_logits(model, images, image_size, num_classes, np_anchor_boxes):
    logits = model(images, training=False)

    true_values_list = [None, None, None]

    scaled_size = image_size / anchors_gen.DOWNSAMPLE_RATIO
    num_boxes = 3

    anchors_reshaped = tf.reshape(np_anchor_boxes, [3, -1])

    pred_bboxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    pred_objs_list = []

    for output_idx, (true_values, pred_values) in enumerate(zip(true_values_list, logits)):
        output_size = int(scaled_size) * tf.math.pow(2, output_idx)

        pred_values = tf.reshape(pred_values, [-1, output_size, output_size, num_boxes, 4 + 1 + num_classes])

        obj_conf = tf.math.sigmoid(pred_values[..., 4])
        tf.print('obj_max:', tf.reduce_max(obj_conf, axis=[1,2,3]))

        pred_bboxes = pred_values[..., 0:4]
        pred_scores_all = tf.math.sigmoid(pred_values[..., 5:])
        pred_scores = tf.reduce_max(pred_scores_all, axis=-1)
        pred_labels = tf.argmax(pred_scores_all, axis=-1)

        pred_labels = tf.reshape(pred_labels, [tf.shape(pred_bboxes)[0], -1])
        pred_labels_list.append(pred_labels)
        pred_scores = tf.reshape(pred_scores, [tf.shape(pred_bboxes)[0], -1])
        pred_scores_list.append(pred_scores)
        pred_objs = tf.reshape(obj_conf, [tf.shape(pred_bboxes)[0], -1])
        pred_objs_list.append(pred_objs)

        anchors_wh = anchors_reshaped[output_idx, :]
        anchors = tf.reshape(anchors_wh, shape=[1, 1, 1, 3, 2])
        pred_box_wh = tf.math.exp(pred_bboxes[..., 2:4]) * anchors

        grid_offset = loss._create_mesh_xy(tf.shape(pred_bboxes)[0], output_size, output_size, num_boxes)
        pred_box_xy = grid_offset + tf.sigmoid(pred_bboxes[..., :2])

        pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)

        pred_bboxes = tf.reshape(pred_bboxes, [tf.shape(pred_bboxes)[0], -1, 4])
        #logger.info('pred_labels: {}, pred_box_xy: {}, pred_box_wh: {}, pred_bboxes: {}'.format(pred_labels.shape, pred_box_xy.shape, pred_box_wh.shape, pred_bboxes.shape))
        pred_bboxes_list.append(pred_bboxes)

    #tf.print('bboxes:', pred_bboxes_list, ', scores:', pred_scores_list, ', labels:', pred_labels_list)

    pred_bboxes = tf.concat(pred_bboxes_list, axis=1)
    pred_labels = tf.concat(pred_labels_list, axis=1)
    pred_scores = tf.concat(pred_scores_list, axis=1)
    pred_objs = tf.concat(pred_objs_list, axis=1)

    #logger.info('pred_bboxes: {}, pred_labels: {}'.format(pred_bboxes.shape, pred_labels.shape))

    return tf.map_fn(lambda out: per_image_supression(out, num_classes),
                     (pred_bboxes, pred_scores, pred_labels, pred_objs),
                     parallel_iterations=FLAGS.num_cpus,
                     back_prop=False,
                     infer_shape=False,
                     dtype=(tf.float32, tf.float32, tf.int32))

def run_eval(model, dataset, num_images, image_size, num_classes, dst_dir, np_anchor_boxes):
    num_files = 0
    for filenames, images in dataset:
        coords_batch, scores_batch, cat_ids_batch = eval_step_logits(model, images, image_size, num_classes, np_anchor_boxes)
        #logger.info('batch: shapes: coords: {}, scores: {}, cat_ids: {}'.format(coords_batch.shape, scores_batch.shape, cat_ids_batch.shape))

        num_files += len(filenames)

        for filename, image, coords, scores, cat_ids in zip(filenames, images, coords_batch, scores_batch, cat_ids_batch):
            filename = str(filename.numpy(), 'utf8')

            good_scores = np.count_nonzero((scores.numpy() > 0))
            logger.info('{}: scores: {}'.format(filename, good_scores))

            anns = []

            for coord, score, cat_id in zip(coords, scores, cat_ids):
                if score.numpy() == 0:
                    break

                coord = coord.numpy()

                cx, cy, w, h = coord
                x0 = cx - w/2
                x1 = cx + w/2
                y0 = cy - h/2
                y1 = cy + h/2

                bb = [x0, y0, x1, y1]

                cat_id = cat_id.numpy()
                area = h * w

                logger.info('{}: bbox: {} -> {}, score: {:.3f}, area: {:.1f}, cat_id: {}'.format(
                    filename,
                    coord, bb, score, area, cat_id))

                anns.append((bb, cat_id))

            r, g, b = tf.split(image, 3, axis=-1)
            image = tf.concat([b, g, r], axis=-1)


            image = image.numpy() * 128 + 128
            image = image.astype(np.uint8)

            cat_names = {}
            dst_filename = os.path.basename(filename)
            dst_filename = os.path.join(FLAGS.output_dir, dst_filename)
            image_draw.draw_im(image, anns, dst_filename, cat_names)

        return
            #for cat_id in range(num_classes)

def run_inference():
    num_classes = FLAGS.num_classes
    num_images = len(FLAGS.filenames)

    if FLAGS.eval_coco_annotations:
        eval_base = coco.create_coco_iterable(FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, logger)

    dtype = tf.float32
    model = yolo.create_model(num_classes)
    np_anchor_boxes, np_anchor_areas, image_size = yolo.create_anchors()

    checkpoint = tf.train.Checkpoint(model=model)

    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    status = checkpoint.restore(checkpoint_prefix)
    status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    if FLAGS.eval_coco_annotations:
        coco.complete_initialization(eval_base, image_size, anchors_boxes, anchor_areas, False)

        num_images = len(eval_base)
        num_classes = eval_base.num_classes()
        cat_names = eval_base.cat_names()

        ds = map_iter.from_indexable(eval_base,
                num_parallel_calls=FLAGS.num_cpus,
                output_types=(tf.string, tf.int64, tf.float32, tf.float32, tf.int32),
                output_shapes=(
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    tf.TensorShape([image_size, image_size, 3]),
                    tf.TensorShape([num_anchors, 4]),
                    tf.TensorShape([num_anchors]),
                ))
        ds = ds.map(lambda filename, image_id, image, true_bboxes, true_labels:
                    tf_left_needed_dimensions(image_size, filename, image_id, image, true_bboxes, true_labels),
                num_parallel_calls=FLAGS.num_cpus)
    else:
        if FLAGS.dataset_type == 'files':
            ds = tf.data.Dataset.from_tensor_slices((FLAGS.filenames))
            ds = ds.map(lambda fn: tf_read_image(fn, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
        elif FLAGS.dataset_type == 'tfrecords':
            from detection import unpack_tfrecord

            filenames = []
            for fn in os.listdir(FLAGS.eval_tfrecord_dir):
                fn = os.path.join(FLAGS.eval_tfrecord_dir, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
            ds = ds.map(lambda record: unpack_tfrecord(record, np_anchor_boxes, np_anchor_areas, image_size, num_classes, False, False, FLAGS.data_format), num_parallel_calls=16)
            ds = ds.map(lambda filename, image_id, image, true_values: tf_left_needed_dimensions_from_tfrecord(image_size, num_classes, np_anchor_boxes, filename, image_id, image, true_values), num_parallel_calls=16)

    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    logger.info('Dataset has been created: num_images: {}, num_classes: {}, model_name: {}'.format(num_images, num_classes, FLAGS.model_name))

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    run_eval(model, ds, num_images, image_size, FLAGS.num_classes, FLAGS.output_dir, np_anchor_boxes)

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
