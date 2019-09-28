import argparse
import cv2
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

logger = logging.getLogger('detection')


import anchors_gen
import coco
import image as image_draw
import loss
import map_iter
import preprocess
import preprocess_ssd
import yolo

parser = argparse.ArgumentParser()
parser.add_argument('--train_coco_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--train_coco_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--eval_coco_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--eval_coco_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run.')
parser.add_argument('--epoch', type=int, default=0, help='Initial epoch\'s number')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs.')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored.')
parser.add_argument('--base_checkpoint', type=str, help='Load base model weights from this file')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
parser.add_argument('--steps_per_eval', default=-1, type=int, help='Number of steps per evaluation run')
parser.add_argument('--print_per_train_steps', default=100, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--steps_per_epoch', default=-1, type=int, help='Number of steps per training run')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--negative_positive_rate', default=2, type=float, help='Negative to positive anchors ratio')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--dataset_type', type=str, choices=['files', 'tfrecords'], default='tfrecords', help='Dataset type')
parser.add_argument('--train_tfrecord_dir', type=str, help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset')
parser.add_argument('--orig_images', action='store_true', help='Whether tfrecords contain original unscaled images')
parser.add_argument('--train_num_images', type=int, help='Number of images in train epoch')
parser.add_argument('--eval_num_images', type=int, help='Number of images in eval epoch')

def unpack_tfrecord(serialized_example, np_anchor_boxes, np_anchor_areas, image_size, num_classes, is_training, orig_images, data_format):
    features = tf.io.parse_single_example(serialized_example,
            features={
                'image_id': tf.io.FixedLenFeature([], tf.int64),
                'filename': tf.io.FixedLenFeature([], tf.string),
                'true_labels': tf.io.FixedLenFeature([], tf.string),
                'true_bboxes': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            })

    orig_bboxes = tf.io.decode_raw(features['true_bboxes'], tf.float32)
    orig_bboxes = tf.reshape(orig_bboxes, [-1, 4])

    orig_labels = tf.io.decode_raw(features['true_labels'], tf.int32)
    filename = features['filename']
    image_id = features['image_id']
    image = tf.image.decode_jpeg(features['image'], channels=3)

    cx, cy, h, w = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image, tf.cast((mx - orig_image_height) / 2, tf.int32), tf.cast((mx - orig_image_width) / 2, tf.int32), mx_int, mx_int)
    cx += (mx - orig_image_width) / 2
    cy += (mx - orig_image_height) / 2

    # convert to XYWH format
    orig_bboxes = tf.concat([cx, cy, w, h], axis=1)

    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    xminf = (cx - w/2) / image_width
    xmaxf = (cx + w/2) / image_width
    yminf = (cy - h/2) / image_height
    ymaxf = (cy + h/2) / image_height

    if orig_images:
        coords_yx = tf.concat([yminf, xminf, ymaxf, xmaxf], axis=1)

        if is_training:
            image, new_labels, new_bboxes = preprocess_ssd.preprocess_for_train(image, orig_labels, coords_yx, [image_size, image_size], data_format=data_format)
            yminf, xminf, ymaxf, xmaxf = tf.split(new_bboxes, num_or_size_splits=4, axis=1)

            orig_labels = new_labels
        else:
            image = preprocess_ssd.preprocess_for_eval(image, [image_size, image_size], data_format=data_format)
    else:
        image = tf.cast(image, tf.float32)
        image -= 128.
        image /= 128.

        image = tf.image.resize_with_pad(image, image_size, image_size)

    cx = (xminf + xmaxf) * image_size / 2
    cy = (yminf + ymaxf) * image_size / 2
    h = (ymaxf - yminf) * image_size
    w = (xmaxf - xminf) * image_size

    orig_bboxes = tf.concat([cx, cy, w, h], axis=1)

    true_values = anchors_gen.generate_true_labels_for_anchors(orig_bboxes, orig_labels, np_anchor_boxes, np_anchor_areas, image_size, num_classes)

    return filename, image_id, image, true_values

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1), square_loss, absolute_loss-0.5)

def draw_bboxes(image_size, train_dataset, train_cat_names, np_anchor_boxes):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    if False:
        for filename, image_id, image, true_values in train_dataset.take(20):
            filename = str(filename.numpy(), 'utf8')

            dst = '{}/{}.png'.format(data_dir, image_id.numpy())
            new_anns = []

            true_values_combined = true_values.numpy()

            bboxes = true_values[:, 0:4]
            labels = true_values[:, 5:]
            labels = np.argmax(labels, axis=1)

            for bb, cat_id in zip(bboxes, labels):
                cx, cy, w, h = bb
                x0 = cx - w/2
                x1 = cx + w/2
                y0 = cy - h/2
                y1 = cy + h/2

                bb = [x0, y0, x1, y1]
                new_anns.append((bb, cat_id))

            #logger.info('{}: true anchors: {}'.format(dst, new_anns))
            logger.info('{}: true anchors: {}'.format(dst, len(new_anns)))

            image = image.numpy() * 128. + 128
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_draw.draw_im(image, new_anns, dst, train_cat_names)

    if True:
        scaled_size = image_size / anchors_gen.DOWNSAMPLE_RATIO
        output_splits = []
        output_sizes = []
        output_indexes = []
        offset = 0
        for base_scale in range(3):
            output_indexes.append(base_scale)
            output_size = scaled_size * np.math.pow(2, base_scale)
            output_sizes.append(output_size)
            offset += int(output_size * output_size)
            output_splits.append(offset)

        for filename, image_id, image, true_values in train_dataset.unbatch().take(20):
            filename = str(filename.numpy(), 'utf8')

            dst = '{}/{}.png'.format(data_dir, image_id.numpy())
            new_anns = []

            true_values_combined = true_values.numpy()
            for true_values, output_size, output_idx in zip(np.split(true_values_combined, output_splits), output_sizes, output_indexes):
                non_background_index_tuple = np.where(true_values[:, :, 4] != 0)
                logger.info('{}: true_values: {}, non_background_index_tuple: {}'.format(filename, true_values.shape, non_background_index_tuple))

                anchor_loc_index = non_background_index_tuple[0]
                box_index = non_background_index_tuple[1]
                if len(anchor_loc_index) == 0:
                    continue

                logger.info('{}: true_values: {}, anchor_loc_index: {}, box_index: {}'.format(filename, true_values.shape, anchor_loc_index, box_index))

                bboxes = true_values[anchor_loc_index, box_index, 0:4]
                labels = true_values[anchor_loc_index, box_index, 5:]
                labels = np.argmax(labels, axis=1)

                logger.info('{}: true_values: {}, bboxes: {}, anchor_loc_index: {}, box_index: {}, labels: {}'.format(
                    filename, true_values.shape, bboxes, anchor_loc_index, box_index, labels))


                for bb, cat_id, bidx in zip(bboxes, labels, box_index):
                    cx, cy, w, h = bb
                    cx = cx / output_size * image_size
                    cy = cy / output_size * image_size

                    anchor_box_idx = output_idx * 3 + bidx
                    anchor = np_anchor_boxes[anchor_box_idx]
                    anchor_w, anchor_h = anchor

                    h = np.math.exp(h) * anchor_h
                    w = np.math.exp(w) * anchor_w

                    x0 = cx - w/2
                    x1 = cx + w/2
                    y0 = cy - h/2
                    y1 = cy + h/2

                    bb = [x0, y0, x1, y1]
                    new_anns.append((bb, cat_id))

            #logger.info('{}: true anchors: {}'.format(dst, new_anns))
            logger.info('{}: true anchors: {}'.format(dst, len(new_anns)))

            image = image.numpy() * 128. + 128
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_draw.draw_im(image, new_anns, dst, train_cat_names)

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    #logger.info('threads: inter(between): {}, intra(within): {}'.format(tf.config.threading.get_inter_op_parallelism_threads(), tf.config.threading.get_intra_op_parallelism_threads()))
    #tf.config.threading.set_inter_op_parallelism_threads(10)
    #tf.config.threading.set_intra_op_parallelism_threads(10)

    num_replicas = 1
    #dstrategy = None
    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
    #if True:
        FLAGS.initial_learning_rate *= num_replicas
        FLAGS.batch_size *= num_replicas

        logdir = os.path.join(FLAGS.train_dir, 'logs')
        writer = tf.summary.create_file_writer(logdir)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        global_step = tf.Variable(1, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

        train_base = coco.create_coco_iterable(FLAGS.train_coco_annotations, FLAGS.train_coco_data_dir, logger)
        eval_base = coco.create_coco_iterable(FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, logger)

        num_classes = train_base.num_classes()
        if num_classes != FLAGS.num_classes:
            num_classes = FLAGS.num_classes

        model = yolo.create_model(num_classes)
        np_anchor_boxes, np_anchor_areas, image_size = yolo.create_anchors()

        def create_dataset(name, base, is_training):
            coco.complete_initialization(base, image_size, np_anchor_boxes, np_anchor_areas, is_training)

            num_images = len(base)
            num_classes = base.num_classes()
            cat_names = base.cat_names()

            ds = map_iter.from_indexable(base,
                    num_parallel_calls=FLAGS.num_cpus,
                    output_types=(tf.string, tf.int64, tf.float32, tf.float32, tf.int32),
                    output_shapes=(
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([image_size, image_size, 3]),
                        tf.TensorShape([num_anchors, 4]),
                        tf.TensorShape([num_anchors]),
                    ))

            if is_training:
                ds = ds.shuffle(200)

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, images: {}, classes: {}'.format(name, num_images, num_classes))

            return ds, num_images, num_classes, cat_names

        def create_dataset_from_tfrecord(name, dataset_dir, image_size, num_classes, is_training):
            filenames = []
            for fn in os.listdir(dataset_dir):
                fn = os.path.join(dataset_dir, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
            ds = ds.map(lambda record: unpack_tfrecord(record, np_anchor_boxes, np_anchor_areas, image_size, num_classes, is_training, FLAGS.orig_images, FLAGS.data_format),
                    num_parallel_calls=FLAGS.num_cpus)
            if is_training:
                ds = ds.shuffle(200)

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, tfrecords: {}'.format(name, len(filenames)))

            return ds

        if FLAGS.dataset_type == 'files':
            train_dataset, train_num_images, train_num_classes, train_cat_names = create_dataset('train', train_base, is_training=True)
            eval_dataset, eval_num_images, eval_num_classes, eval_cat_names = create_dataset('eval', eval_base, is_training=False)
        elif FLAGS.dataset_type == 'tfrecords':
            train_num_images = FLAGS.train_num_images
            eval_num_images = FLAGS.eval_num_images
            train_num_classes = FLAGS.num_classes
            train_cat_names = {}

            train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, image_size, train_num_classes, is_training=True)
            eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, image_size, train_num_classes, is_training=False)


        if False:
            draw_bboxes(image_size, train_dataset, train_cat_names, np_anchor_boxes)
            exit(0)

        if train_num_classes is None:
            logger.error('If there is no train_num_classes (tfrecord dataset), you must provide --num_classes')
            exit(-1)

        if train_num_images is None:
            logger.error('If there is no train_num_images (tfrecord dataset), you must provide --117266,')
            exit(-1)
        if eval_num_images is None:
            logger.error('If there is no eval_num_images (tfrecord dataset), you must provide --steps_per_eval')
            exit(-1)

        steps_per_epoch = calc_epoch_steps(train_num_images)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch

        steps_per_eval = calc_epoch_steps(eval_num_images)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch (used/calc): {}/{}, train images: {}, steps_per_eval (used/calc): {}/{}, eval images: {}'.format(
            steps_per_epoch, calc_epoch_steps(train_num_images), train_num_images,
            steps_per_eval, calc_epoch_steps(eval_num_images), eval_num_images))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}".format(manager.latest_checkpoint, global_step.numpy()))
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

            if FLAGS.base_checkpoint:
                base_checkpoint = tf.train.Checkpoint(model=model.base_model)
                status = base_checkpoint.restore(FLAGS.base_checkpoint)
                status.expect_partial()

                logger.info("Restored base model from external checkpoint {}".format(FLAGS.base_checkpoint))

        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.Accuracy(name='train_accuracy')
        obj_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_objectness_accuracy')
        iou_metric = tf.keras.metrics.Mean(name='train_iou')
        num_good_ious_metric = tf.keras.metrics.Mean(name='eval_num_good_ious')
        num_positive_labels_metric = tf.keras.metrics.Mean(name='eval_num_positive_labels_ious')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = tf.keras.metrics.Accuracy(name='eval_accuracy')
        eval_obj_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='eval_objectness_accuracy')
        eval_distance_center_metric = tf.keras.metrics.MeanAbsoluteError(name='eval_dist_center')
        eval_distance_size_metric = tf.keras.metrics.MeanAbsoluteError(name='eval_dist_size')
        eval_iou_metric = tf.keras.metrics.Mean(name='eval_iou')
        eval_num_good_ious_metric = tf.keras.metrics.Mean(name='eval_num_good_ious')
        eval_num_positive_labels_metric = tf.keras.metrics.Mean(name='eval_num_positive_labels_ious')

        def reset_metrics():
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            obj_accuracy_metric.reset_states()
            iou_metric.reset_states()
            num_good_ious_metric.reset_states()
            num_positive_labels_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()
            eval_obj_accuracy_metric.reset_states()
            eval_iou_metric.reset_states()
            eval_num_good_ious_metric.reset_states()
            eval_num_positive_labels_metric.reset_states()

        scaled_size = image_size / anchors_gen.DOWNSAMPLE_RATIO
        output_sizes, output_splits = [], []
        offset = 0
        num_scales = 3
        num_boxes = 3
        for base_scale in range(num_scales):
            output_size = scaled_size * np.math.pow(2, base_scale)
            output_sizes.append(int(output_size))
            output_splits.append(int(output_size * output_size))

        yolo_loss_object = loss.YOLOLoss(image_size, np_anchor_boxes, output_sizes, reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(logits, true_values, loss_metric, accuracy_metric, obj_accuracy_metric, iou_metric, num_good_ious_metric, num_positive_labels_metric):
            true_values_list = list(tf.split(true_values, output_splits, axis=1))
            dist_loss, class_loss, conf_loss_pos, conf_loss_neg = yolo_loss_object.call(y_true_list=true_values_list, y_pred_list=logits)

            dist_loss = tf.nn.compute_average_loss(dist_loss, global_batch_size=FLAGS.batch_size)
            class_loss = tf.nn.compute_average_loss(class_loss, global_batch_size=FLAGS.batch_size)
            conf_loss_pos = tf.nn.compute_average_loss(conf_loss_pos, global_batch_size=FLAGS.batch_size)
            conf_loss_neg = tf.nn.compute_average_loss(conf_loss_neg, global_batch_size=FLAGS.batch_size)

            total_loss = dist_loss + class_loss + conf_loss_pos + conf_loss_neg
            loss_metric.update_state(total_loss)

            anchors_reshaped = tf.reshape(np_anchor_boxes, [3, -1])

            for output_idx, (true_values, pred_values) in enumerate(zip(true_values_list, logits)):
                output_size = int(scaled_size) * tf.math.pow(2, output_idx)

                #logger.info('true_values: {}, pred_values: {}'.format(true_values.shape, pred_values.shape))

                true_values = tf.reshape(true_values, [-1, output_size, output_size, num_boxes, 4 + 1 + num_classes])
                pred_values = tf.reshape(pred_values, [-1, output_size, output_size, num_boxes, 4 + 1 + num_classes])

                non_background_index = tf.where(tf.not_equal(true_values[..., 4], 0))
                num_positive_labels_metric.update_state(tf.shape(non_background_index)[0])
                #tf.print('num_positive_labels_metric:', tf.math.count_nonzero(true_values[..., 4]))

                box_index = non_background_index[:, 3]

                #logger.info('non_background_index: {}'.format(non_background_index.shape))
                #tf.print('non_background_index:', non_background_index)
                #tf.print('box_index:', box_index)

                sampled_true_values = tf.gather_nd(true_values, non_background_index)
                sampled_pred_values = tf.gather_nd(pred_values, non_background_index)

                true_obj = sampled_true_values[:, 5]
                true_bboxes = sampled_true_values[:, 0:4]
                true_labels = sampled_true_values[:, 5:]
                true_labels = tf.argmax(true_labels, axis=1)

                pred_obj = tf.math.sigmoid(sampled_pred_values[:, 5])
                pred_bboxes = sampled_pred_values[:, 0:4]
                pred_labels = sampled_pred_values[:, 5:]
                pred_labels = tf.argmax(pred_labels, axis=1)

                #logger.info('true_bboxes: {}, pred_bboxes: {}'.format(true_bboxes.shape, pred_bboxes.shape))
                #logger.info('true_labels: {}, pred_labels: {}'.format(true_labels.shape, pred_labels.shape))

                #tf.print('true_labels:', true_labels, ', pred_labels:', pred_labels)

                accuracy_metric.update_state(pred_labels, true_labels)
                obj_accuracy_metric.update_state(y_true=true_obj, y_pred=pred_obj)

                anchors_wh = anchors_reshaped[output_idx, :]

                grid_offset = loss._create_mesh_xy(tf.shape(true_bboxes)[0], output_size, output_size, 3)
                grid_offset = tf.gather_nd(grid_offset, non_background_index)

                anchor_grid = loss._create_mesh_anchor(anchors_wh, tf.shape(true_bboxes)[0], output_size, output_size, 3)
                anchor_grid = tf.gather_nd(anchor_grid, non_background_index)

                pred_box_xy = grid_offset + tf.math.sigmoid(pred_bboxes[..., :2])

                anchors_wh = tf.reshape(anchors_wh, [3, 2])
                anchors_wh = tf.expand_dims(anchors_wh, 0)
                anchors_wh = tf.tile(anchors_wh, [tf.shape(pred_box_xy)[0], 1, 1])

                box_index = tf.one_hot(box_index, 3, dtype=tf.float32)
                box_index = tf.expand_dims(box_index, -1)

                #logger.info('anchors_wh: {}, box_index: {}'.format(anchors_wh.shape, box_index.shape))
                anchors_wh = tf.reduce_sum(anchors_wh * box_index, axis=1)
                #logger.info('anchors_wh: {}'.format(anchors_wh.shape))

                pred_box_wh = tf.math.exp(pred_bboxes[..., 2:4]) * anchors_wh

                #logger.info('pred_box_xy: {}, pred_box_wh: {}, anchors_wh: {}'.format(pred_box_xy.shape, pred_box_wh.shape, anchors_wh.shape))

                # true_bboxes contain upper left corner of the box
                true_xy = true_bboxes[..., 0:2] * tf.cast(output_size, tf.float32) / float(image_size)
                true_wh = tf.math.exp(true_bboxes[..., 2:4]) * anchor_grid

                #logger.info('true_xy: {}, true_wh: {}, true_wh_orig: {}, anchor_grid: {}'.format(true_xy.shape, true_wh.shape, true_bboxes[..., 2:4].shape, anchor_grid.shape))

                pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
                true_bboxes = tf.concat([true_xy, true_wh], axis=-1)
                ious = anchors_gen.calc_ious_one_to_one(pred_bboxes, true_bboxes)
                #tf.print('pred_bboxes_shape:', tf.shape(pred_bboxes), 'true_bboxes_shape:', tf.shape(true_bboxes))
                #tf.print('ious:', ious)
                iou_metric.update_state(ious)

                good_ious = tf.where(ious > 0.5)
                num_good_ious_metric.update_state(tf.shape(good_ious)[0])

            return dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        def eval_step(filenames, images, true_values):
            logits = model(images, training=False)
            dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss = calculate_metrics(logits, true_values,
                    eval_loss_metric, eval_accuracy_metric, eval_obj_accuracy_metric, eval_iou_metric,
                    eval_num_good_ious_metric, eval_num_positive_labels_metric)
            return dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss

        def train_step(filenames, images, true_values):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss = calculate_metrics(logits, true_values,
                        loss_metric, accuracy_metric, obj_accuracy_metric, iou_metric,
                        num_good_ious_metric, num_positive_labels_metric)

            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)

            stddev = 1 / ((1 + epoch_var)**0.55)

            clip_gradients = []
            for g, v in zip(gradients, variables):
                if g is None:
                    logger.info('no gradients for variable: {}'.format(v))
                else:
                    g += tf.random.normal(stddev=stddev, mean=0., shape=g.shape)
                    #g = tf.clip_by_value(g, -5, 5)

                clip_gradients.append(g)
            opt.apply_gradients(zip(clip_gradients, variables))

            global_step.assign_add(1)

            return dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss

        @tf.function
        def distributed_train_step(args):
            dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss = dstrategy.experimental_run_v2(train_step, args=args)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, dist_loss, axis=None)
            class_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, class_loss, axis=None)
            conf_loss_pos = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_pos, axis=None)
            conf_loss_neg = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_neg, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss

        @tf.function
        def distributed_eval_step(args):
            dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss = dstrategy.experimental_run_v2(eval_step, args=args)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, dist_loss, axis=None)
            class_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, class_loss, axis=None)
            conf_loss_pos = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_pos, axis=None)
            conf_loss_neg = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_neg, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, image_ids, images, true_values in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss = step_func(args=(filenames, images, true_values))
                if name == 'train' and step % FLAGS.print_per_train_steps == 0:
                    logger.info('{}: {}: step: {}/{}, dist_loss: {:.2e}, class_loss: {:.2e}, conf_loss: {:.2e}/{:.2e}, total_loss: {:.2e}, accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/pos: {}/{}'.format(
                        name, int(epoch_var.numpy()), step, max_steps,
                        dist_loss, class_loss, conf_loss_pos, conf_loss_neg, total_loss,
                        accuracy_metric.result(), obj_accuracy_metric.result(),
                        iou_metric.result(),
                        int(num_good_ious_metric.result()), int(num_positive_labels_metric.result()),
                        ))

                step += 1
                if step >= max_steps:
                    break

            return step

        min_metric = 0
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier

        if False:
            with writer.as_default():
                train_steps = run_epoch('train', train_dataset, distributed_train_step, 1)
                from tensorflow.python.keras import backend as K
                from tensorflow.python.ops import summary_ops_v2
                from tensorflow.python.eager import context

                with context.eager_mode():
                    with summary_ops_v2.always_record_summaries():
                        if not model.run_eagerly:
                            summary_ops_v2.graph(K.get_graph(), step=0)

                        summary_writable = (
                                model._is_graph_network or  # pylint: disable=protected-access
                                model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
                        if summary_writable:
                            summary_ops_v2.keras_model('keras', model, step=0)

                exit(0)

        def validation_metric():
            eval_acc = eval_accuracy_metric.result()
            eval_iou = eval_iou_metric.result()
            metric = eval_acc + eval_iou

            return metric

        if manager.latest_checkpoint:
            reset_metrics()
            logger.info('there is a checkpoint {}, running initial validation'.format(manager.latest_checkpoint))

            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval)
            min_metric = validation_metric()
            logger.info('initial validation metric: {:.3f}'.format(min_metric))

        if min_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {:.3f} -> {} from command line arguments'.format(min_metric, FLAGS.min_eval_metric))
            min_metric = FLAGS.min_eval_metric

        num_vars = len(model.trainable_variables)
        num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, model trainable variables/params: {}/{}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
            num_vars, int(num_params)))

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            epoch_var.assign(epoch)

            reset_metrics()

            train_steps = run_epoch('train', train_dataset, distributed_train_step, steps_per_epoch)
            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval)

            metric = validation_metric()

            logger.info('epoch: {}, train: steps: {}, accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/pos: {}/{}, loss: {:.2e}, eval: accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/pos: {}/{}, loss: {:.2e}, lr: {:.2e}, val_metric: {:.3f}'.format(
                epoch, global_step.numpy(),
                accuracy_metric.result(), obj_accuracy_metric.result(), iou_metric.result(),
                int(num_good_ious_metric.result()), int(num_positive_labels_metric.result()),
                loss_metric.result(),
                eval_accuracy_metric.result(), eval_obj_accuracy_metric.result(), eval_iou_metric.result(),
                int(eval_num_good_ious_metric.result()), int(eval_num_positive_labels_metric.result()),
                eval_loss_metric.result(),
                learning_rate.numpy(),
                metric))

            if metric > min_metric:
                save_path = manager.save()
                logger.info("epoch: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}, accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/positive: {}/{}".format(
                    epoch, save_path, min_metric, metric, 
                    eval_accuracy_metric.result(), eval_obj_accuracy_metric.result(), eval_iou_metric.result(),
                    int(eval_num_good_ious_metric.result()), int(eval_num_positive_labels_metric.result())))
                min_metric = metric
                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                want_reset = False

                if learning_rate > FLAGS.min_learning_rate:
                    new_lr = learning_rate.numpy() * learning_rate_multiplier
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, min_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    if learning_rate_multiplier > 0.1:
                        learning_rate_multiplier /= 2

                    want_reset = True
                elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                    new_lr = FLAGS.initial_learning_rate
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, min_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    want_reset = True
                    learning_rate_multiplier = initial_learning_rate_multiplier

                if want_reset:
                    logger.info('epoch: {}, best metric: {:.5f}, learning rate: {:.2e}, restoring best checkpoint: {}'.format(
                        epoch, min_metric, learning_rate.numpy(), manager.latest_checkpoint))

                    checkpoint.restore(manager.latest_checkpoint)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    FLAGS = parser.parse_args()

    try:
        train()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
