import argparse
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf

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
        anchor_boxes_for_layers = anchor.create_anchors(image_size, feature_shapes)

        def create_dataset(name, ann_file, data_dir, is_training):
            coco_iterable = coco.create_coco_iterable(image_size, ann_file, data_dir, logger, is_training, anchor_boxes_for_layers)

            num_images = len(coco_iterable)
            num_classes = coco_iterable.num_classes()
            cat_names = coco_iterable.cat_names()
            num_anchors = 0
            for shape, anchors in anchor_boxes_for_layers:
                logger.info('layer: {}, anchors: {}'.format(shape, len(anchors)))
                num_anchors += len(anchors)

            ds = map_iter.from_indexable(coco_iterable,
                    num_parallel_calls=FLAGS.num_cpus,
                    output_types=(tf.string, tf.int64, tf.float32, tf.float32, tf.int32),
                    output_shapes=(
                        tf.TensorShape([]),
                        tf.TensorShape([]),
                        tf.TensorShape([image_size, image_size, 3]),
                        tf.TensorShape([num_anchors, 4]),
                        tf.TensorShape([num_anchors]),
                    ))

            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            logger.info('{} dataset has been created, data dir: {}, is_training: {}, images: {}, classes: {}, anchors: {}'.format(
                name, ann_file, is_training, num_images, num_classes, num_anchors))

            return ds, num_images, num_classes, cat_names

        train_dataset, train_num_images, train_num_classes, train_cat_names = create_dataset('train', FLAGS.train_coco_annotations, FLAGS.train_coco_data_dir, is_training=True)

        if False:
            data_dir = os.path.join(FLAGS.train_dir, 'tmp')
            os.makedirs(data_dir, exist_ok=True)

            for filename, image_id, image, true_bboxes, true_labels in train_dataset.take(10):
                filename = str(filename)
                dst = '{}/{}.png'.format(data_dir, image_id.numpy())
                new_anns = []
                for bb, cat_id in zip(true_bboxes, true_labels):
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

        base_model, model, image_size, feature_shapes = ssd.create_model(params, dtype, FLAGS.model_name, train_num_classes)

        dummy_input = tf.ones((int(FLAGS.batch_size / num_replicas), image_size, image_size, 3), dtype=dtype)
        dstrategy.experimental_run_v2(call_base_model, args=(base_model, dummy_input))
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
        accuracy_metric = loss.IOUScore(threshold=0.5, name='train_accuracy')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = loss.IOUScore(threshold=0.5, name='eval_accuracy')

        def reset_metrics():
            loss_metric.reset_states()
            accuracy_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()

        #cross_entropy_loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, label_smoothing=FLAGS.label_smoothing)
        cross_entropy_loss_object = loss.CategoricalFocalLoss(reduction=tf.keras.losses.Reduction.NONE)
        dice_loss = loss.DiceLoss(reduction=tf.keras.losses.Reduction.NONE, class_weights=class_weights)

        def calculate_metrics(logits, labels):
            #one_hot_labels = tf.one_hot(labels, num_classes, axis=3)
            #one_hot_labels = tf.squeeze(one_hot_labels, axis=4)

            #ce_loss = cross_entropy_loss_object(y_pred=logits, y_true=one_hot_labels)
            ce_loss = cross_entropy_loss_object(y_pred=logits, y_true=labels)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

            total_loss = ce_loss

            if True:
                de_loss = dice_loss(y_pred=logits, y_true=labels)
                de_loss = tf.nn.compute_average_loss(de_loss, global_batch_size=FLAGS.batch_size)
                de_loss = 1. - de_loss
                total_loss += de_loss

            return total_loss

        def eval_step(filenames, images, labels):
            logits = model(images, training=False)
            total_loss = calculate_metrics(logits, labels)

            eval_loss_metric.update_state(total_loss)
            eval_accuracy_metric.update_state(labels, logits)

        def train_step(filenames, images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                total_loss = calculate_metrics(logits, labels)

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

            loss_metric.update_state(total_loss)
            accuracy_metric.update_state(labels, logits)

            global_step.assign_add(1)

            return total_loss

        @tf.function
        def distributed_train_step(args):
            pr_total_losses = dstrategy.experimental_run_v2(train_step, args=args)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)
            return total_loss

        @tf.function
        def distributed_eval_step(args):
            return dstrategy.experimental_run_v2(eval_step, args=args)

        def run_epoch(dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, images, masks in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                step_func(args=(filenames, images, masks))

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

            eval_steps = run_epoch(eval_dataset, distributed_eval_step, steps_per_eval)
            min_metric = eval_accuracy_metric.result()
            logger.info('initial validation metric: {}'.format(min_metric))

        if min_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {} -> {} from command line arguments'.format(min_metric, FLAGS.min_eval_metric))
            min_metric = FLAGS.min_eval_metric

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            reset_metrics()

            train_steps = run_epoch(train_dataset, distributed_train_step, steps_per_epoch)
            eval_steps = run_epoch(eval_dataset, distributed_eval_step, steps_per_eval)

            logger.info('epoch: {}, train: steps: {}, loss: {:.2e}, accuracy: {:.5f}, eval: loss: {:.2e}, accuracy: {:.5f}, lr: {:.2e}'.format(
                epoch, global_step.numpy(),
                loss_metric.result(), accuracy_metric.result(),
                eval_loss_metric.result(), eval_accuracy_metric.result(),
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

            if num_epochs_without_improvement >= 10:
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
                elif num_epochs_without_improvement >= 10:
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
    #np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True)

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
