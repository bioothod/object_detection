import argparse
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import anchor
import coco
import image as image_draw
import loss
import map_iter
import validate
import ssd

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

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
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
parser.add_argument('--steps_per_eval', default=-1, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_epoch', default=-1, type=int, help='Number of steps per training run')
parser.add_argument('--min_eval_metric', default=0.75, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--negative_positive_rate', default=2, type=float, help='Negative to positive anchors ratio')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
FLAGS = parser.parse_args()

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def local_swish(x):
    return x * tf.nn.sigmoid(x)

@tf.function
def call_base_model(model, inputs):
    #return model(inputs, training=True, features_only=True)
    return model(inputs, training=True)

@tf.function
def call_model(model, inputs):
    return model(inputs, training=True)

def run_eval(model, eval_dataset, eval_num_images, train_dir, global_step):
    @tf.function
    def eval_step_mask(images):
        logits = model(images, training=False)
        return logits

    dst_dir = os.path.join(train_dir, 'eval', str(global_step))
    os.makedirs(dst_dir, exist_ok=True)

    num_files = 0
    for filenames, images, true_masks in eval_dataset:
        masks = eval_step_mask(images)
        num_files += len(filenames)

        validate.generate_images(filenames, images, masks, dst_dir)
        logger.info('saved {}/{} images'.format(len(filenames), num_files))

        if num_files >= eval_num_images:
            break

    logger.info('global_step: {}, evaluated {} images -> {}'.format(global_step, num_files, dst_dir))

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

#    logger.info('threads: inter(between): {}, intra(within): {}'.format(tf.config.threading.get_inter_op_parallelism_threads(), tf.config.threading.get_intra_op_parallelism_threads()))
#    tf.config.threading.set_inter_op_parallelism_threads(10)
#    tf.config.threading.set_intra_op_parallelism_threads(10)

    num_replicas = 1
    #dstrategy = None
    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
    #if True:
        FLAGS.initial_learning_rate *= num_replicas
        FLAGS.batch_size *= num_replicas

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        global_step = tf.Variable(1, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

        base_model, image_size, feature_shapes = ssd.create_base_model(dtype, FLAGS.model_name)

        np_anchor_boxes, np_anchor_areas, anchor_layers = anchor.create_anchors(image_size, feature_shapes)
        num_anchors = np_anchor_boxes.shape[0]
        logger.info('base model: {}, num_anchors: {}'.format(FLAGS.model_name, num_anchors))

        def create_dataset(name, ann_file, data_dir, is_training):
            coco_iterable = coco.create_coco_iterable(image_size, ann_file, data_dir, logger, is_training, np_anchor_boxes, np_anchor_areas)

            num_images = len(coco_iterable)
            num_classes = coco_iterable.num_classes()
            cat_names = coco_iterable.cat_names()

            ds = map_iter.from_indexable(coco_iterable,
                    num_parallel_calls=FLAGS.num_cpus,
                    output_types=(tf.string, tf.int64, tf.float32, tf.float32, tf.int32, tf.int32),
                    output_shapes=(
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([image_size, image_size, 3]),
                        tf.TensorShape([num_anchors, 4]),
                        tf.TensorShape([num_anchors]),
                        tf.TensorShape([num_anchors]),
                    ))

            if is_training:
                ds = ds.shuffle(FLAGS.batch_size * 2)

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, data dir: {}, is_training: {}, images: {}, classes: {}, anchors: {}'.format(
                name, ann_file, is_training, num_images, num_classes, num_anchors))

            return ds, num_images, num_classes, cat_names

        train_dataset, train_num_images, train_num_classes, train_cat_names = create_dataset('train', FLAGS.train_coco_annotations, FLAGS.train_coco_data_dir, is_training=True)

        if False:
            data_dir = os.path.join(FLAGS.train_dir, 'tmp')
            os.makedirs(data_dir, exist_ok=True)

            for filename, image_id, image, true_bboxes, true_labels, true_orig_labels in train_dataset.unbatch().take(10):
                filename = str(filename)
                dst = '{}/{}.png'.format(data_dir, image_id.numpy())
                new_anns = []
                # category ID in true_labels does not match train_cat_names here, this is converted category_id into range [0, num_categories], where 0 is background class
                # true_orig_labels contain original category ids
                for bb, cat_id in zip(true_bboxes, true_orig_labels):
                    new_anns.append((bb.numpy(), cat_id.numpy()))

                image = image.numpy() * 128. + 128
                image = image.astype(np.uint8)
                image_draw.draw_im(image, new_anns, dst, train_cat_names)

            exit(0)

        eval_dataset, eval_num_images, eval_num_classes, eval_cat_names = create_dataset('eval', FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, is_training=False)

        steps_per_epoch = calc_epoch_steps(train_num_images)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_num_images)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch: {}/{}, train images: {}, steps_per_eval: {}/{}, eval images: {}'.format(
            steps_per_epoch, calc_epoch_steps(train_num_images), train_num_images,
            steps_per_eval, calc_epoch_steps(eval_num_images), eval_num_images))

        model = ssd.create_model(base_model, image_size, train_num_classes, anchor_layers, feature_shapes)

        dummy_input = tf.ones((int(FLAGS.batch_size / num_replicas), image_size, image_size, 3), dtype=dtype)
        dstrategy.experimental_run_v2(call_model, args=(model, dummy_input))

        num_vars_base = 0
        num_vars_base = len(base_model.trainable_variables)
        num_params_base = 0
        num_params_base = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])

        num_vars_ssd = 0
        num_vars_ssd = len(model.trainable_variables)
        num_params_ssd = 0
        num_params_ssd = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, base model trainable variables/params: {}/{}, ssd model trainable variables/params: {}/{}, dtype: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
            num_vars_base, int(num_params_base),
            num_vars_ssd, int(num_params_ssd),
            dtype))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        if FLAGS.checkpoint:
            status = checkpoint.restore(FLAGS.checkpoint)
            status.expect_partial()

            logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))

        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}".format(manager.latest_checkpoint, global_step.numpy()))
            #run_eval(model, eval_dataset, eval_num_images, FLAGS.train_dir, global_step.numpy())
            #exit(0)
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        distance_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
        iou_metric = loss.IOUScore(threshold=0.5, name='train_iou')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
        eval_distance_metric = tf.keras.metrics.MeanAbsoluteError(name='eval_mae')
        eval_iou_metric = loss.IOUScore(threshold=0.5, name='eval_iou')

        def reset_metrics():
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            distance_metric.reset_states()
            iou_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()
            eval_distance_metric.reset_states()
            eval_iou_metric.reset_states()

        ce_loss_object = loss.CategoricalFocalLoss(reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(logits, bboxes, true_labels, loss_metric, accuracy_metric, distance_metric):
            prev_anchors = 0
            total_mae_loss = None
            total_ce_loss = None
            num_positives = []
            for (coords, classes), num_anchor_boxes_for_layer, shape in zip(logits, anchor_layers, feature_shapes):
                num_anchors_per_output = int(num_anchor_boxes_for_layer / (shape[1] * shape[1]))

                if coords.shape[1:3] != shape[1:3]:
                    logger.error('layer: {}, anchors: {}, coords: {}, classes: {}'.format(shape, num_anchor_boxes_for_layer, coords, classes))
                    exit(-1)

                orig_coords_shape = coords.shape
                orig_classes_shape = classes.shape

                coords = tf.reshape(coords, [-1, num_anchor_boxes_for_layer, 4])
                classes = tf.reshape(classes, [-1, num_anchor_boxes_for_layer, train_num_classes])

                bboxes_for_layer = bboxes[:, prev_anchors:prev_anchors + num_anchor_boxes_for_layer, :]
                true_labels_for_layer = true_labels[:, prev_anchors:prev_anchors + num_anchor_boxes_for_layer]

                logger.debug('base_layer: {}, anchors: {}, coords: {} -> {}, classes: {} -> {}, bboxes_for_layer: {}, true_labels_for_layer: {}'.format(
                    shape, num_anchor_boxes_for_layer,
                    orig_coords_shape, coords.shape,
                    orig_classes_shape, classes.shape,
                    bboxes_for_layer.shape, true_labels_for_layer.shape))

                prev_anchors += num_anchor_boxes_for_layer

                positive_indexes = tf.where(true_labels_for_layer > 0)
                num_positive = tf.shape(positive_indexes)[0]
                num_positives.append(num_positive)

                num_negative = tf.cast(FLAGS.negative_positive_rate * tf.cast(num_positive, tf.float32), tf.int32)

                # because 'x == 0' condition yields (none, 0) tensor, and ' < 1' yields (none, 2) shape, the same as aboe positive index selection
                negative_indexes = tf.where(true_labels_for_layer < 1)
                negative_indexes = tf.random.shuffle(negative_indexes)
                negative_indexes = negative_indexes[:num_negative]

                new_indexes = tf.concat([positive_indexes, negative_indexes], axis=0)

                sampled_true_labels = tf.gather_nd(true_labels_for_layer, new_indexes)
                sampled_true_labels_one_hot = tf.one_hot(sampled_true_labels, train_num_classes, axis=-1)
                sampled_pred_classes = tf.gather_nd(classes, new_indexes)

                sampled_true_bboxes = tf.gather_nd(bboxes_for_layer, new_indexes)
                sampled_pred_coords = tf.gather_nd(coords, new_indexes)

                ce_loss = ce_loss_object(y_true=sampled_true_labels_one_hot, y_pred=sampled_pred_classes)
                ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

                mae_loss = tf.keras.losses.MAE(y_true=sampled_true_bboxes, y_pred=sampled_pred_coords)
                mae_loss = tf.nn.compute_average_loss(mae_loss, global_batch_size=FLAGS.batch_size)

                sampled_true_labels = tf.gather_nd(true_labels_for_layer, positive_indexes)
                sampled_pred_classes = tf.gather_nd(classes, positive_indexes)
                accuracy_metric.update_state(y_true=sampled_true_labels, y_pred=sampled_pred_classes)
                distance_metric.update_state(y_true=sampled_true_bboxes, y_pred=sampled_pred_coords)

                if total_ce_loss is None:
                    total_ce_loss = ce_loss
                    total_mae_loss = mae_loss
                else:
                    total_ce_loss += ce_loss
                    total_mae_loss += mae_loss


            total_mae_loss = total_mae_loss *  0.01

            total_loss = total_ce_loss + total_mae_loss
            loss_metric.update_state(total_loss)

            return total_ce_loss, total_mae_loss, total_loss, num_positives

        def eval_step(filenames, images, bboxes, true_labels):
            logits = model(images, training=False)
            ce_loss, mae_loss, total_loss, num_positive = calculate_metrics(logits, bboxes, true_labels, eval_loss_metric, eval_accuracy_metric, eval_distance_metric)
            return ce_loss, mae_loss, total_loss, num_positive

        def train_step(filenames, images, bboxes, true_labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                ce_loss, mae_loss, total_loss, num_positive = calculate_metrics(logits, bboxes, true_labels, loss_metric, accuracy_metric, distance_metric)

            #variables = model.trainable_variables + base_model.trainable_variables
            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            clip_gradients = []
            for g, v in zip(gradients, variables):
                if g is None:
                    logger.info('no gradients for variable: {}'.format(v))
                else:
                    g = tf.clip_by_value(g, -1, 1)

                clip_gradients.append(g)
            opt.apply_gradients(zip(clip_gradients, variables))

            global_step.assign_add(1)

            return ce_loss, mae_loss, total_loss, num_positive

        @tf.function
        def distributed_train_step_dummy(args):
            ret = []
            for x in dstrategy.experimental_run_v2(train_step, args=args):
                x = dstrategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)
                ret.append(x)

            return ret

        @tf.function
        def distributed_train_step(args):
            pr_ce_losses, pr_mae_losses, pr_total_losses, num_positive = dstrategy.experimental_run_v2(train_step, args=args)
            ce_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_ce_losses, axis=None)
            mae_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_mae_losses, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)
            num_positive = dstrategy.reduce(tf.distribute.ReduceOp.SUM, num_positive, axis=0)
            return ce_loss, mae_loss, total_loss, num_positive

        @tf.function
        def distributed_eval_step(args):
            pr_ce_losses, pr_mae_losses, pr_total_losses, num_positive = dstrategy.experimental_run_v2(eval_step, args=args)
            ce_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_ce_losses, axis=None)
            mae_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_mae_losses, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)
            num_positive = dstrategy.reduce(tf.distribute.ReduceOp.SUM, num_positive, axis=0)
            return ce_loss, mae_loss, total_loss, num_positive

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, image_ids, images, bboxes, true_labels, orig_true_labels in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                ce_loss, mae_loss, total_loss, num_positive = step_func(args=(filenames, images, bboxes, true_labels))
                if name == 'train':
                    logger.info('{}: step: {}, num_positive: {}, ce_loss: {:.2e}, mae_loss: {:.2e}, total_loss: {:.2e}, accuracy: {:.4f}, distance: {:.2f}'.format(
                        name, step, num_positive, ce_loss, mae_loss, total_loss, accuracy_metric.result(), loss_metric.result()))

                step += 1
                if step >= max_steps:
                    break

            return step

        min_metric = 0
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier

        if manager.latest_checkpoint:
            reset_metrics()
            logger.info('there is a checkpoint {}, running initial validation'.format(manager.latest_checkpoint))

            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval)
            min_metric = eval_accuracy_metric.result()
            logger.info('initial validation metric: {:.5f}'.format(min_metric))

        if min_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {:.5f} -> {} from command line arguments'.format(min_metric, FLAGS.min_eval_metric))
            min_metric = FLAGS.min_eval_metric

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            reset_metrics()

            train_steps = run_epoch('train', train_dataset, distributed_train_step, steps_per_epoch)
            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval)

            logger.info('epoch: {}, train: steps: {}, accuracy: {:.4f}, distance: {:.2e}, loss: {:.2e}, eval: accuracy: {:.4f}, distance: {:.2e}, loss: {:.2e}, lr: {:.2e}'.format(
                epoch, global_step.numpy(),
                accuracy_metric.result(), distance_metric.result(), loss_metric.result(),
                eval_accuracy_metric.result(), eval_distance_metric.result(), eval_loss_metric.result(),
                learning_rate.numpy()))

            eval_acc = eval_accuracy_metric.result()
            if eval_acc > min_metric:
                save_path = manager.save()
                logger.info("epoch: {}, saved checkpoint: {}, eval accuracy: {:.5f} -> {:.5f}".format(epoch, save_path, min_metric, eval_acc))
                min_metric = eval_acc
                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

                if False:
                    run_eval(model, eval_dataset, eval_num_images, FLAGS.train_dir, global_step.numpy())
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
