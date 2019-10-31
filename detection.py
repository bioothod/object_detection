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
import encoder
import image as image_draw
import loss
import preprocess_ssd
import yolo

parser = argparse.ArgumentParser()
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
parser.add_argument('--dataset_type', type=str, choices=['tfrecords'], default='tfrecords', help='Dataset type')
parser.add_argument('--train_tfrecord_dir', type=str, help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--image_size', type=int, default=0, help='Use this image size, if 0 - use default')
parser.add_argument('--train_num_images', type=int, help='Number of images in train epoch')
parser.add_argument('--eval_num_images', type=int, help='Number of images in eval epoch')

def unpack_tfrecord(serialized_example, anchors_all, output_xy_grids, output_ratios, image_size, is_training, data_format):
    features = tf.io.parse_single_example(serialized_example,
            features={
                'image_id': tf.io.FixedLenFeature([], tf.int64),
                'filename': tf.io.FixedLenFeature([], tf.string),
                'true_labels': tf.io.FixedLenFeature([], tf.string),
                'true_bboxes': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            })
    filename = features['filename']

    orig_bboxes = tf.io.decode_raw(features['true_bboxes'], tf.float32)
    orig_bboxes = tf.reshape(orig_bboxes, [-1, 4])

    true_text_labels = tf.strings.split(features['true_labels'], '<SEP>')

    image_id = features['image_id']
    image = tf.image.decode_jpeg(features['image'], channels=3)

    cx, cy, h, w = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2, tf.int32),
                tf.cast((mx - orig_image_width) / 2, tf.int32),
                mx_int,
                mx_int)
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

    coords_yx = tf.concat([yminf, xminf, ymaxf, xmaxf], axis=1)

    if is_training:
        image, new_text_labels, new_bboxes = preprocess_ssd.preprocess_for_train(image, true_text_labels, coords_yx,
                [image_size, image_size], data_format=data_format)

        yminf, xminf, ymaxf, xmaxf = tf.split(new_bboxes, num_or_size_splits=4, axis=1)
    else:
        image = preprocess_ssd.preprocess_for_eval(image, [image_size, image_size], data_format=data_format)
        new_text_labels = true_text_labels

    cx = (xminf + xmaxf) * image_size / 2
    cy = (yminf + ymaxf) * image_size / 2
    h = (ymaxf - yminf) * image_size
    w = (xmaxf - xminf) * image_size

    new_bboxes = tf.concat([cx, cy, w, h], axis=1)

    true_values, true_texts = anchors_gen.generate_true_values_for_anchors(new_bboxes, new_text_labels,
            anchors_all, output_xy_grids, output_ratios,
            image_size)

    return filename, image_id, image, true_values, true_texts

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def draw_bboxes(image_size, train_dataset, all_anchors, all_grid_xy, all_ratios):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    all_anchors = all_anchors.numpy()
    all_grid_xy = all_grid_xy.numpy()
    all_ratios = all_ratios.numpy()

    for filename, image_id, image, true_values, true_texts in train_dataset.unbatch().take(20):
        filename = str(filename.numpy(), 'utf8')

        dst = '{}/{}.png'.format(data_dir, image_id.numpy())

        true_values = true_values.numpy()
        true_texts = true_texts.numpy()

        non_background_index = np.where(true_values[..., 4] != 0)[0]
        #logger.info('{}: true_values: {}, non_background_index: {}'.format(filename, true_values.shape, non_background_index.shape))

        labels = true_texts[non_background_index]
        bboxes = true_values[non_background_index, 0:4]

        anchors = all_anchors[non_background_index, :]
        grid_xy = all_grid_xy[non_background_index, :]
        ratios = all_ratios[non_background_index]

        cx, cy, w, h = np.split(bboxes, 4, axis=1)
        cx = np.squeeze(cx)
        cy = np.squeeze(cy)
        w = np.squeeze(w)
        h = np.squeeze(h)

        #logger.info('bboxes: {}, grid_xy: {}, anchors: {}, ratios: {}'.format(bboxes, grid_xy, anchors, ratios))
        #logger.info('cx: {}, cy: {}, w: {}, h: {}'.format(cx, cy, w, h))

        cx = (cx + grid_xy[:, 0]) * ratios
        cy = (cy + grid_xy[:, 1]) * ratios

        #logger.info('cx: {}, anchors: {}'.format(cx, anchors))
        w = np.power(np.math.e, w) * anchors[:, 2]
        h = np.power(np.math.e, h) * anchors[:, 3]

        x0 = cx - w/2
        x1 = cx + w/2
        y0 = cy - h/2
        y1 = cy + h/2

        bb = np.stack([x0, y0, x1, y1], axis=1)
        new_anns = []
        for _bb, l in zip(bb, labels):
            l = str(l, 'utf8')
            new_anns.append((_bb, None, l))

        logger.info('{}: true anchors: {}'.format(dst, len(new_anns)))

        image = preprocess_ssd.denormalize_image(image)
        image = image.numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_draw.draw_im(image, new_anns, dst, {})

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    good_checkpoint_dir = os.path.join(checkpoint_dir, 'good')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(good_checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
        FLAGS.initial_learning_rate *= num_replicas
        FLAGS.batch_size *= num_replicas

        logdir = os.path.join(FLAGS.train_dir, 'logs')
        writer = tf.summary.create_file_writer(logdir)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        global_step = tf.Variable(1, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

        image_size = FLAGS.image_size
        model = encoder.create_model(FLAGS.model_name)
        image_size = model.image_size
        if model.output_sizes is None:
            dummy_input = tf.ones((int(FLAGS.batch_size / num_replicas), image_size, image_size, 3), dtype=dtype)
            dstrategy.experimental_run_v2(lambda m, inp: m(inp, True), args=(model, dummy_input))
            logger.info('model output sizes: {}'.format(model.output_sizes))

        anchors_all, output_xy_grids, output_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

        def create_dataset_from_tfrecord(name, dataset_dir, image_size, is_training):
            filenames = []
            for fn in os.listdir(dataset_dir):
                fn = os.path.join(dataset_dir, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
            ds = ds.map(lambda record: unpack_tfrecord(record, anchors_all, output_xy_grids, output_ratios,
                            image_size, is_training,
                            FLAGS.data_format),
                    num_parallel_calls=FLAGS.num_cpus)
            if is_training:
                ds = ds.shuffle(200)

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, tfrecords: {}'.format(name, len(filenames)))

            return ds

        if FLAGS.dataset_type == 'tfrecords':
            train_num_images = FLAGS.train_num_images
            eval_num_images = FLAGS.eval_num_images

            train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, image_size, is_training=True)
            eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, image_size, is_training=False)


        if False:
            draw_bboxes(image_size, train_dataset, anchors_all, output_xy_grids, output_ratios)
            exit(0)

        if train_num_images is None:
            logger.error('If there is no train_num_images (tfrecord dataset), you must provide --train_num_images')
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
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}".format(manager.latest_checkpoint, global_step.numpy()))
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

            if FLAGS.base_checkpoint:
                base_checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model.body)
                status = base_checkpoint.restore(FLAGS.base_checkpoint)
                status.expect_partial()

                saved_path = manager.save()
                logger.info("Restored base model from external checkpoint {} and saved object-based checkpoint {}".format(FLAGS.base_checkpoint, saved_path))
                exit(0)

        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        obj_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_objectness_accuracy')
        iou_metric = tf.keras.metrics.Mean(name='train_best_iou')
        num_good_ious_metric = tf.keras.metrics.Mean(name='eval_num_good_ious')
        num_positive_labels_metric = tf.keras.metrics.Mean(name='eval_num_positive_labels_ious')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='eval_accuracy')
        eval_obj_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='eval_objectness_accuracy')
        eval_iou_metric = tf.keras.metrics.Mean(name='eval_best_iou')
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

        ylo = loss.YOLOLoss(anchors_all, output_xy_grids, output_ratios, image_size, reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(images, is_training, true_values,
                loss_metric, accuracy_metric, obj_accuracy_metric,
                iou_metric, num_good_ious_metric, num_positive_labels_metric):

            obj_logits = model(images, training=is_training)
            dist_loss, conf_loss_pos, conf_loss_neg = ylo.call(y_true=true_values, y_pred=obj_logits)

            dist_loss = tf.nn.compute_average_loss(dist_loss, global_batch_size=FLAGS.batch_size)
            conf_loss_pos = tf.nn.compute_average_loss(conf_loss_pos, global_batch_size=FLAGS.batch_size)
            conf_loss_neg = tf.nn.compute_average_loss(conf_loss_neg, global_batch_size=FLAGS.batch_size)

            total_loss = dist_loss + conf_loss_pos + conf_loss_neg
            loss_metric.update_state(total_loss)

            y_true = true_values
            y_pred = obj_logits

            object_mask = tf.expand_dims(y_true[..., 4], -1)
            object_mask_bool = tf.cast(object_mask[..., 0], 'bool')
            num_positive_labels_metric.update_state(tf.math.count_nonzero(object_mask))

            #sigmoid(t_xy) + c_xy
            pred_box_xy = ylo.grid_xy + tf.sigmoid(y_pred[..., :2])
            pred_box_xy = pred_box_xy * ylo.ratios
            pred_box_wh = tf.math.exp(y_pred[..., 2:4]) * ylo.anchors_wh

            # confidence/objectiveness
            true_obj_mask = tf.boolean_mask(object_mask, object_mask_bool)
            pred_box_conf = tf.boolean_mask(tf.expand_dims(y_pred[..., 4], -1), object_mask_bool)
            pred_box_conf = tf.math.sigmoid(pred_box_conf)
            obj_accuracy_metric.update_state(y_true=true_obj_mask, y_pred=pred_box_conf)

            true_xy = (y_true[..., 0:2] + ylo.grid_xy) * ylo.ratios
            true_wh = tf.math.exp(y_true[..., 2:4]) * ylo.anchors_wh

            pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
            true_bboxes = tf.concat([true_xy, true_wh], axis=-1)

            def get_best_ious(input_tuple, object_mask, true_bboxes):
                idx, pred_boxes_for_single_image = input_tuple
                valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
                # shape: [N, 4] & [V, 4] ==> [N, V]
                ious = loss.box_iou(pred_boxes_for_single_image, valid_true_boxes)
                # shape: [N, V] -> [V]
                best_ious = tf.reduce_max(ious, axis=[0])
                best_ious_padded = tf.pad(best_ious,
                        [[0, tf.cast(tf.math.count_nonzero(object_mask), tf.int32) - tf.shape(valid_true_boxes)[0]]],
                        constant_values=-1.)
                return best_ious_padded

            best_ious = tf.map_fn(lambda t: get_best_ious(t, object_mask, true_bboxes),
                                (tf.range(tf.shape(pred_bboxes)[0]), pred_bboxes),
                                parallel_iterations=32,
                                back_prop=False,
                                dtype=(tf.float32))

            best_ious = tf.reshape(best_ious, [-1])
            best_ious_index = tf.where(best_ious >= 0)
            best_ious = tf.gather_nd(best_ious, best_ious_index)
            iou_metric.update_state(best_ious)

            good_ious = tf.where(best_ious > 0.5)
            num_good_ious_metric.update_state(tf.shape(good_ious)[0])

            return dist_loss, conf_loss_pos, conf_loss_neg, total_loss

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        def eval_step(filenames, images, true_values, true_text_labels):
            dist_loss, conf_loss_pos, conf_loss_neg, total_loss = calculate_metrics(images, False, true_values,
                    eval_loss_metric, eval_accuracy_metric, eval_obj_accuracy_metric, eval_iou_metric,
                    eval_num_good_ious_metric, eval_num_positive_labels_metric)
            return dist_loss, conf_loss_pos, conf_loss_neg, total_loss

        def train_step(filenames, images, true_values, true_text_labels):
            with tf.GradientTape() as tape:
                dist_loss, conf_loss_pos, conf_loss_neg, total_loss = calculate_metrics(images, True, true_values,
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
                    #g += tf.random.normal(stddev=stddev, mean=0., shape=g.shape)
                    #g = tf.clip_by_value(g, -5, 5)
                    pass

                clip_gradients.append(g)
            opt.apply_gradients(zip(clip_gradients, variables))

            global_step.assign_add(1)

            return dist_loss, conf_loss_pos, conf_loss_neg, total_loss

        @tf.function
        def distributed_train_step(args):
            dist_loss, conf_loss_pos, conf_loss_neg, total_loss = dstrategy.experimental_run_v2(train_step, args=args)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, dist_loss, axis=None)
            conf_loss_pos = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_pos, axis=None)
            conf_loss_neg = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_neg, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return dist_loss, conf_loss_pos, conf_loss_neg, total_loss

        @tf.function
        def distributed_eval_step(args):
            dist_loss, conf_loss_pos, conf_loss_neg, total_loss = dstrategy.experimental_run_v2(eval_step, args=args)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, dist_loss, axis=None)
            conf_loss_pos = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_pos, axis=None)
            conf_loss_neg = dstrategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss_neg, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return dist_loss, conf_loss_pos, conf_loss_neg, total_loss

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, image_ids, images, true_values, true_text_labels in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                dist_loss, conf_loss_pos, conf_loss_neg, total_loss = step_func(args=(filenames, images, true_values, true_text_labels))
                if (name == 'train' and step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                    logger.info('{}: {}: step: {}/{}, dist_loss: {:.2e}, conf_loss: {:.2e}/{:.2e}, total_loss: {:.2e}, accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/pos: {}/{}'.format(
                        name, int(epoch_var.numpy()), step, max_steps,
                        dist_loss, conf_loss_pos, conf_loss_neg, total_loss,
                        accuracy_metric.result(), obj_accuracy_metric.result(),
                        iou_metric.result(),
                        int(num_good_ious_metric.result()), int(num_positive_labels_metric.result()),
                        ))

                    if np.isnan(total_loss.numpy()):
                        exit(-1)


                step += 1
                if step >= max_steps:
                    break

            return step

        min_metric = 0
        best_saved_path = None
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier

        def validation_metric():
            eval_acc = eval_accuracy_metric.result()
            eval_obj_acc = eval_obj_accuracy_metric.result()
            eval_iou = eval_iou_metric.result()
            metric = eval_acc + eval_iou + eval_obj_acc*2

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

            saved_path = manager.save()

            if metric > min_metric:
                best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, metric))

                logger.info("epoch: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}, accuracy: {:.3f}, obj_acc: {:.3f}, iou: {:.3f}, good_ios/positive: {}/{}".format(
                    epoch, best_saved_path, min_metric, metric,
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

                    #want_reset = True
                elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                    new_lr = FLAGS.initial_learning_rate
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, min_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    want_reset = True
                    learning_rate_multiplier = initial_learning_rate_multiplier

                if want_reset and best_saved_path is not None:
                    logger.info('epoch: {}, best metric: {:.5f}, learning rate: {:.2e}, restoring best checkpoint: {}'.format(
                        epoch, min_metric, learning_rate.numpy(), best_saved_path))

                    checkpoint.restore(best_saved_path)


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
