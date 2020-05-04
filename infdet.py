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

import coco
import encoder
import image as image_draw
import preprocess_ssd

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_image_height = tf.cast(tf.shape(image)[0], dtype)
    orig_image_width = tf.cast(tf.shape(image)[1], dtype)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image, tf.cast((mx - orig_image_height) / 2, tf.int32), tf.cast((mx - orig_image_width) / 2, tf.int32), mx_int, mx_int)

    image = preprocess_ssd.preprocess_for_eval(image, [image_size, image_size], data_format=FLAGS.data_format)

    return filename, image

def tf_left_needed_dimensions_from_tfrecord(image_size, anchors_all, output_xy_grids, output_ratios, num_classes, filename, image_id, image, true_values):
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

def eval_step_logits(model, images, image_size, num_classes, all_anchors, min_score, min_size, max_ret, iou_threshold):
    bboxes, categories, scores = model(images, training=False)
    h, w = images.shape[1: 3]

    # Iterate through images in batch, and for each one
    # create the ground truth coco annotation

    for batch_idx in range(len(bboxes)):
        gt_labels, gt_boxes = true_labels[batch_idx], true_bboxes[batch_idx]
        no_padding_mask = gt_labels != -1

        gt_labels = tf.boolean_mask(gt_labels, no_padding_mask)
        gt_boxes = tf.boolean_mask(gt_boxes, no_padding_mask)

        im_annot, annots = _COCO_gt_annot(image_id, annot_id, (h, w), gt_labels, gt_boxes)
        gt_coco['annotations'].extend(annots)
        gt_coco['images'].append(im_annot)

        preds = categories[batch_idx], bboxes[batch_idx], scores[batch_idx]
        pred_labels, pred_boxes, pred_scores = preds

        if pred_labels.shape[0] > 0:
            results = _COCO_result(image_id, pred_labels, pred_boxes, pred_scores)
            results_coco.extend(results)

        annot_id += len(annots)
        image_id += 1

    total_time += time.time() - inference_start
    num_images += len(bboxes)

    if i % print_every == 0:
        logger.info('validated steps: {}/{}, images: {}, perf: {:.1f} img/s, time_per_image: {:.1f} ms'.format(
            i, steps, num_images, num_images / total_time, total_time / num_images * 1000))

def run_eval(model, dataset, num_images, image_size, num_classes, dst_dir, cat_names, FLAGS):
    num_files = 0
    dump_js = []
    for filenames, images in dataset:
        start_time = time.time()
        coords_batch, scores_batch, cat_ids_batch = model(images, training=False, score_threshold=FLAGS.min_score, iou_threshold=FLAGS.iou_threshold, max_ret=FLAGS.max_ret)

        num_files += len(filenames)
        time_per_image_ms = (time.time() - start_time) / len(filenames) * 1000

        #logger.info('bboxes: {}, scores: {}, labels: {}'.format(coords_batch.shape, scores_batch.shape, cat_ids_batch.shape))

        logger.info('batch images: {}, total_processed: {}, time_per_image: {:.1f} ms'.format(len(filenames), num_files, time_per_image_ms))

        for filename, image, coords, scores, cat_ids in zip(filenames, images, coords_batch, scores_batch, cat_ids_batch):
            filename = str(filename.numpy(), 'utf8')
            image = cv2.imread(filename)

            base_filename = os.path.basename(filename)

            imh, imw = image.shape[:2]
            max_side = max(imh, imw)
            pad_y = (max_side - imh) / 2
            pad_x = (max_side - imw) / 2
            square_scale = max_side / image_size

            scores = scores.numpy()
            cat_ids = cat_ids.numpy()


            anns = []

            js = {
                'filename': filename,
            }
            js_anns = []

            for coord, score, cat_id in zip(coords, scores, cat_ids):
                score = float(score)
                cat_id = int(cat_id)

                if score <= 0:
                    break

                bb = coord.numpy().astype(float)
                scaled_bb = bb * square_scale

                xmin, ymin, xmax, ymax = scaled_bb
                xmin -= pad_x
                xmax -= pad_x
                ymin -= pad_y
                ymax -= pad_y

                #ymin, xmin, ymax, xmax = bb
                bb = [xmin, ymin, xmax, ymax]

                ann_js = {
                    'bbox': bb,
                    'class_score': score,
                    'category_id': cat_id,
                }

                js_anns.append(ann_js)
                anns.append((bb, None, cat_id))

                #logger.info('{}: bbox: {}, obj: {:.3f}, score: {:.3f}, cat_id: {}'.format(filename, bb, objs, score, cat_id))

            js['annotations'] = js_anns
            dump_js.append(js)

            if not FLAGS.do_not_save_images:
                if not FLAGS.no_channel_swap:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                dst_filename = os.path.basename(filename)
                dst_filename = os.path.join(FLAGS.output_dir, dst_filename)
                image_draw.draw_im(image, anns, dst_filename, cat_names)

    json_fn = os.path.join(FLAGS.output_dir, 'results.json')
    logger.info('Saving {} objects into {}'.format(len(dump_js), json_fn))

    with open(json_fn, 'w') as fout:
        json.dump(dump_js, fout, indent=2)


def run_inference(FLAGS):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, 'infdet.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    num_classes = FLAGS.num_classes
    num_images = len(FLAGS.filenames)

    dtype = tf.float32
    model = encoder.create_model(FLAGS.d, FLAGS.num_classes)
    image_size = model.config.input_size

    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model)

    status = checkpoint.restore(checkpoint_prefix)
    if FLAGS.skip_checkpoint_assertion:
        status.expect_partial()
    else:
        status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    with open(FLAGS.category_json, 'r') as f:
        class2idx = json.load(f)

    cat_names = {}
    for cname, cid in class2idx.items():
        cat_names[cid] = cname

    if FLAGS.eval_coco_annotations:
        eval_base = coco.create_coco_iterable(FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, logger)

        num_images = len(eval_base)
        num_classes = eval_base.num_classes()
        cat_names = eval_base.cat_names()

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

        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=8)
        ds = ds.map(lambda record: unpack_tfrecord(record, all_anchors, all_grid_xy, all_ratios,
                    image_size, num_classes, False,
                    FLAGS.data_format, FLAGS.do_not_step_labels),
            num_parallel_calls=16)
        ds = ds.map(lambda filename, image_id, image, true_values: tf_left_needed_dimensions_from_tfrecord(image_size, all_anchors, all_grid_xy, all_ratios, num_classes,
                    filename, image_id, image, true_values),
            num_parallel_calls=16)

    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    logger.info('Dataset has been created: num_images: {}, num_classes: {}, D: {}'.format(num_images, num_classes, FLAGS.d))

    run_eval(model, ds, num_images, image_size, FLAGS.num_classes, FLAGS.output_dir, cat_names, FLAGS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_coco_annotations', type=str, help='Path to MS COCO dataset: annotations json file')
    parser.add_argument('--eval_coco_data_dir', type=str, default='/', help='Path to MS COCO dataset: image directory')
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
    parser.add_argument('--d', type=int, default=0, help='Model name')
    parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
    parser.add_argument('--skip_checkpoint_assertion', action='store_true', help='Skip checkpoint assertion, needed for older models on objdet3/4')
    parser.add_argument('--no_channel_swap', action='store_true', help='When set, do not perform rgb-bgr conversion, needed, when using files as input')
    parser.add_argument('--do_not_save_images', action='store_true', help='Do not save images with bounding boxes')
    parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
    parser.add_argument('--category_json', type=str, help='Category to ID mapping json file.')
    parser.add_argument('--dataset_type', type=str, choices=['files', 'tfrecords', 'filelist'], default='files', help='Dataset type')
    parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')
    FLAGS = parser.parse_args()

    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    if not FLAGS.checkpoint and not FLAGS.checkpoint_dir:
        logger.critical('You must provide either checkpoint or checkpoint dir')
        exit(-1)

    try:
        run_inference(FLAGS)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
