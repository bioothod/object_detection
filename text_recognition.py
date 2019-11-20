import argparse
import cv2
import logging
import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

logger = logging.getLogger('detection')


import autoaugment
import encoder_gated_conv as encoder
import preprocess

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
parser.add_argument('--train_num_images', required=True, type=int, help='Number of images per train epoch')
parser.add_argument('--eval_num_images', required=True, type=int, help='Number of images per eval epoch')
parser.add_argument('--print_per_train_steps', default=100, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--best_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--train_tfrecord_dir', type=str, help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--image_shape', type=str, default='32x128', help='Use this image shape: HxW')

dict_split = tf.strings.unicode_split(encoder.default_char_dictionary, 'UTF-8')
kv_init = tf.lookup.KeyValueTensorInitializer(keys=dict_split, values=tf.range(1, dict_split.shape[0]+1, 1), key_dtype=tf.string, value_dtype=tf.int32)
dict_table = tf.lookup.StaticHashTable(kv_init, 0)
max_text_length = 128

def unpack_tfrecord(serialized_example, image_shape, is_training, data_format):
    features = tf.io.parse_single_example(serialized_example,
            features={
                'image_id': tf.io.FixedLenFeature([], tf.int64),
                'filename': tf.io.FixedLenFeature([], tf.string),
                'text': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            })
    filename = features['filename']

    text = features['text']
    text_split = tf.strings.unicode_split(text, 'UTF-8')

    image_id = features['image_id']
    image = tf.image.decode_jpeg(features['image'], channels=3)

    image = preprocess.preprocess_image(image, image_shape, is_training)

    text_tensor = dict_table.lookup(text_split)
    
    text_tensor = text_tensor[:max_text_length]
    text_length = tf.shape(text_tensor)[0]

    pad_size = tf.maximum(max_text_length - text_length, 0)
    text_tensor = tf.pad(text_tensor, [[0, pad_size]])

    return filename, image_id, image, text_tensor, text_length

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

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

        image_shape = [int(d) for d in FLAGS.image_shape.split('x')][:2]
        logger.info('input shape from command line: {}'.format(image_shape))

        model = encoder.create_text_recognition_model(FLAGS.model_name, max_text_length)

        def create_dataset_from_tfrecord(name, dataset_dir, image_shape, is_training):
            filenames = []
            for fn in os.listdir(dataset_dir):
                fn = os.path.join(dataset_dir, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

            random.shuffle(filenames)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
            ds = ds.map(lambda record: unpack_tfrecord(record,
                            image_shape, is_training,
                            FLAGS.data_format),
                    num_parallel_calls=FLAGS.num_cpus)

            def filter_fn(filename, image_id, image, text_tensor, text_length):
                return tf.math.not_equal(text_tensor[0], 0)

            ds = ds.filter(filter_fn)

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, tfrecords: {}'.format(name, len(filenames)))

            return ds

        train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, image_shape, is_training=True)
        eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, image_shape, is_training=False)

        steps_per_train_epoch = calc_epoch_steps(FLAGS.train_num_images)
        steps_per_eval_epoch = calc_epoch_steps(FLAGS.eval_num_images)

        logger.info('steps_per_train_epoch: {}, train images: {}, steps_per_eval_epoch: {}, eval images: {}'.format(
            steps_per_train_epoch, FLAGS.train_num_images,
            steps_per_eval_epoch, FLAGS.eval_num_images))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}".format(manager.latest_checkpoint, global_step.numpy()))
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

            if FLAGS.base_checkpoint:
                base_checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model.base_model)
                status = base_checkpoint.restore(FLAGS.base_checkpoint)
                status.expect_partial()

                saved_path = manager.save()
                logger.info("Restored base model from external checkpoint {} and saved object-based checkpoint {}".format(FLAGS.base_checkpoint, saved_path))
                exit(0)

        loss_metric = tf.keras.metrics.Mean(name='train_loss')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')

        def reset_metrics():
            loss_metric.reset_states()

            eval_loss_metric.reset_states()

        def calculate_metrics(logits, true_texts, true_lengths, loss_metric):
            sparse = tf.sparse.from_dense(true_texts)

            logit_length = tf.ones_like(true_lengths) * max_text_length
            ctc_loss = tf.nn.ctc_loss(labels=sparse, label_length=None, blank_index=0, logits=logits, logit_length=logit_length, logits_time_major=False)
            ctc_loss = tf.nn.compute_average_loss(ctc_loss, global_batch_size=FLAGS.batch_size)

            total_loss = ctc_loss
            loss_metric.update_state(total_loss)

            return total_loss

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        def eval_step(filenames, images, true_values, true_lengths):
            logits = model(images, training=False)
            total_loss = calculate_metrics(logits, true_values, true_lengths, eval_loss_metric)
            return total_loss

        def train_step(filenames, images, true_values, true_lengths):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                total_loss = calculate_metrics(logits, true_values, true_lengths, loss_metric)

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

            return total_loss

        @tf.function
        def distributed_train_step(args):
            total_loss = dstrategy.experimental_run_v2(train_step, args=args)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.MEAN, total_loss, axis=None)
            return total_loss

        @tf.function
        def distributed_eval_step(args):
            total_loss = dstrategy.experimental_run_v2(eval_step, args=args)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.MEAN, total_loss, axis=None)
            return total_loss

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, image_ids, images, true_texts, true_lengths in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                total_loss = step_func(args=(filenames, images, true_texts, true_lengths))
                if (name == 'train' and step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                    logger.info('{}: {}: step: {}/{}, global_step: {}, total_loss: {:.2e}'.format(
                        name, int(epoch_var.numpy()), step, max_steps, global_step.numpy(), total_loss))

                    if np.isnan(total_loss.numpy()):
                        exit(-1)


                step += 1
                if step >= max_steps:
                    break

            return step

        best_metric = FLAGS.best_eval_metric
        best_saved_path = None
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier
        epoch = global_step.numpy() / steps_per_train_epoch

        learning_rate.assign(FLAGS.initial_learning_rate)

        def validation_metric():
            eval_loss = eval_loss_metric.result()

            return eval_loss

        if manager.latest_checkpoint:
            reset_metrics()
            logger.info('there is a checkpoint {}, running initial validation'.format(manager.latest_checkpoint))

            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval_epoch)
            best_metric = validation_metric()
            logger.info('initial validation metric: {:.3f}'.format(best_metric))

        if best_metric >= FLAGS.best_eval_metric:
            logger.info('setting minimal evaluation metric {:.3f} -> {} from command line arguments'.format(best_metric, FLAGS.best_eval_metric))
            best_metric = FLAGS.best_eval_metric

        num_vars = len(model.trainable_variables)
        num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_shape: {}, model trainable variables/params: {}/{}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_shape,
            num_vars, int(num_params)))

        for epoch in range(int(epoch), FLAGS.num_epochs):
            epoch_var.assign(epoch)

            reset_metrics()

            train_steps = run_epoch('train', train_dataset, distributed_train_step, steps_per_train_epoch)
            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval_epoch)

            metric = validation_metric()

            logger.info('epoch: {}, train: steps: {}, loss: {:.2e}, eval: loss: {:.2e}, lr: {:.2e}, val_metric: {:.3f}'.format(
                epoch, global_step.numpy(),
                loss_metric.result(),
                eval_loss_metric.result(),
                learning_rate.numpy(),
                metric))

            saved_path = manager.save()

            if metric < best_metric:
                best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, metric))

                logger.info("epoch: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}".format(
                    epoch, best_saved_path, best_metric, metric))
                best_metric = metric
                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                want_reset = False

                if learning_rate > FLAGS.min_learning_rate:
                    new_lr = learning_rate.numpy() * learning_rate_multiplier
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    if learning_rate_multiplier > 0.1:
                        learning_rate_multiplier /= 2

                    #want_reset = True
                elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                    new_lr = FLAGS.initial_learning_rate
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    want_reset = True
                    learning_rate_multiplier = initial_learning_rate_multiplier

                if want_reset and best_saved_path is not None:
                    logger.info('epoch: {}, best metric: {:.5f}, learning rate: {:.2e}, restoring best checkpoint: {}'.format(
                        epoch, best_metric, learning_rate.numpy(), best_saved_path))

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
