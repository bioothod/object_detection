import argparse
import cv2
import logging
import os
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import horovod.tensorflow as hvd


import tensorflow_addons as tfa

logger = logging.getLogger('detection')


import anchors_gen
import encoder
import image as image_draw
import loss
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--force_epoch', type=int, help='Force this epoch number')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run')
parser.add_argument('--skip_saving_epochs', type=int, default=0, help='Do not save good checkpoint and update best metric for this number of the first epochs')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored')
parser.add_argument('--base_checkpoint', type=str, help='Load base model weights from this file')
parser.add_argument('--use_good_checkpoint', action='store_true', help='Recover from the last good checkpoint when present')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
parser.add_argument('--print_per_train_steps', default=100, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--dataset_type', type=str, choices=['tfrecords'], default='tfrecords', help='Dataset type')
parser.add_argument('--warmup_tfrecrods_epochs', default=50, type=float, help='Drop tfrecords from --train_tfrecord_dir_warmup after this many epochs')
parser.add_argument('--train_tfrecord_dir_warmup', type=str, action='append', help='Directory containing training TFRecords, which will be dropped after --warmup_tfrecrods_epochs epochs')
parser.add_argument('--train_tfrecord_dir', type=str, required=True, action='append', help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, required=True, action='append', help='Directory containing evaluation TFRecords')
parser.add_argument('--image_size', type=int, required=True, help='Use this image size, if 0 - use default')
parser.add_argument('--crop_size', type=str, default='8x24', help='Use this sizes for feature crops')
parser.add_argument('--steps_per_eval_epoch', default=30, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_train_epoch', default=200, type=int, help='Number of steps per train run')
parser.add_argument('--steps_per_warmup_epoch', default=1000, type=int, help='Number of steps per train warmup run')
parser.add_argument('--save_examples', type=int, default=0, help='Number of example images to save and exit')
parser.add_argument('--max_sequence_len', type=int, default=64, help='Maximum word length, also number of RNN attention timesteps')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--use_predicted_polys_epochs', type=int, default=-1, help='After how many epochs to use predicted polynome coordinates for feature crops, negative means never')
parser.add_argument('--warmup_objdet_epochs', type=int, default=100, help='Start using normal (1.0) objdet loss scale after this epoch, use heavily diminished before that')
parser.add_argument('--max_word_batch', type=int, default=64, help='Maximum batch of word')
parser.add_argument('--disable_rotation_augmentation', action='store_true', help='Whether to disable rotation/flipping augmentation')
parser.add_argument('--use_random_augmentation', action='store_true', help='Use efficientnet random augmentation')

default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
parser.add_argument('--dictionary', type=str, default=default_char_dictionary, help='Dictionary to use')

def unpack_tfrecord(record, anchors_all, image_size, max_sequence_len, dict_table, pad_value, is_training, data_format, dtype):
    features = tf.io.parse_single_example(record,
            features={
                'filename': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
                'true_labels': tf.io.FixedLenFeature([], tf.string),
                'word_poly': tf.io.FixedLenFeature([], tf.string),
            })

    filename = features['filename']

    image = features['image']
    image = tf.image.decode_jpeg(image, channels=3)

    word_poly = tf.io.decode_raw(features['word_poly'], tf.float32)
    word_poly = tf.reshape(word_poly, [-1, 4, 2])

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2., tf.int32),
                tf.cast((mx - orig_image_width) / 2., tf.int32),
                mx_int,
                mx_int)

    xdiff = (mx - orig_image_width) / 2
    ydiff = (mx - orig_image_height) / 2

    add = tf.stack([xdiff, ydiff])
    word_poly += add

    image = tf.image.resize(image, [image_size, image_size])

    scale = image_size / mx
    word_poly *= scale
    word_poly = tf.cast(word_poly, dtype)

    image = tf.cast(image, dtype)
    image -= 128
    image /= 128

    text_labels = tf.strings.split(features['true_labels'], '<SEP>')

    if is_training:
        image, word_poly, text_labels = preprocess.preprocess_for_train(image, word_poly, text_labels, image_size, FLAGS.disable_rotation_augmentation, FLAGS.use_random_augmentation)

    text_split = tf.strings.unicode_split(text_labels, 'UTF-8')

    text_lenghts = text_split.row_lengths()
    text_lenghts = tf.expand_dims(text_lenghts, 1)

    encoded_values = dict_table.lookup(text_split.values)
    rg = tf.RaggedTensor.from_row_splits(values=encoded_values, row_splits=text_split.row_splits)
    encoded_padded_text = rg.to_tensor(default_value=pad_value)
    encoded_padded_text = encoded_padded_text[..., :max_sequence_len]

    to_add = max_sequence_len - tf.shape(encoded_padded_text)[1]
    if to_add > 0:
        encoded_padded_text = tf.pad(encoded_padded_text, [[0, 0], [0, to_add]], mode='CONSTANT', constant_values=pad_value)

    true_values = anchors_gen.generate_true_values_for_anchors(word_poly, anchors_all, encoded_padded_text, text_lenghts, max_sequence_len)

    return filename, image, true_values

def draw_bboxes(image_size, train_dataset, num_examples, all_anchors, dictionary, pad_value, max_sequence_len):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    for filename, image, true_values in train_dataset.unbatch().take(num_examples):
        filename = str(filename.numpy(), 'utf8')
        filename_base = os.path.basename(filename)
        filename_base = os.path.splitext(filename_base)[0]

        dst = os.path.join(data_dir, filename_base) + '.png'

        image = image.numpy()
        image = image * 128 + 128
        image = image.astype(np.uint8)

        true_word_obj, word_poly, text_labels, lengths = anchors_gen.unpack_true_values(true_values, all_anchors, image.shape, image_size, max_sequence_len)

        word_poly = word_poly.numpy()
        text_labels = text_labels.numpy().astype(np.int32)

        new_anns = []
        for poly, text in zip(word_poly, text_labels):
            text_str = ''
            for text_idx in text[0]:
                if text_idx != pad_value and text_idx <= len(dictionary):
                    text_str += dictionary[text_idx - 1]

            #logger.info('{}: {}: {}'.format(filename, text_str, poly))
            new_anns.append((None, poly, text_str))

        image_draw.draw_im(image, new_anns, dst, {})

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    good_checkpoint_dir = os.path.join(checkpoint_dir, 'good')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(good_checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log.{}'.format(hvd.rank())), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    num_replicas = hvd.size()

    dtype = tf.float32
    if FLAGS.use_fp16:
        dtype = tf.float16
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)


    FLAGS.initial_learning_rate *= num_replicas

    if hvd.rank() == 0:
        logdir = os.path.join(FLAGS.train_dir, 'logs')
        writer = tf.summary.create_file_writer(logdir)
        writer.set_as_default()

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

    dictionary_size, dict_table, pad_value = anchors_gen.create_lookup_table(FLAGS.dictionary)

    image_size = FLAGS.image_size
    params = encoder.create_params(FLAGS.model_name, image_size, FLAGS.crop_size, FLAGS.max_sequence_len, dictionary_size, pad_value, dtype)
    model = encoder.Encoder(params)

    dummy_input = tf.ones((FLAGS.batch_size, image_size, image_size, 3), dtype=dtype)

    logits, rnn_features = model(dummy_input, training=True)

    anchors_all, output_xy_grids, output_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes, dtype)

    true_word_obj = logits[..., 0]
    true_word_poly = logits[..., 1 : 9]

    num_anchors = tf.shape(true_word_obj)[1]

    true_words = tf.ones((FLAGS.batch_size, num_anchors, FLAGS.max_sequence_len), dtype=tf.int64)
    true_lengths = tf.ones((FLAGS.batch_size, num_anchors), dtype=tf.int64)

    tidx = tf.range(num_anchors)
    tidx = tf.expand_dims(tidx, 0)
    tidx = tf.tile(tidx, [FLAGS.batch_size, 1])
    true_word_obj = tf.where(tidx < 256 // FLAGS.batch_size, 1, 0)
    test_words = tf.math.count_nonzero(true_word_obj)

    rnn_outputs, rnn_outputs_ar = model.rnn_inference_from_true_values(logits, rnn_features,
                                                                       true_word_obj, true_word_poly, true_words, true_lengths,
                                                                       anchors_all, training=True, use_predicted_polys=True)
    line_length = 128
    model.body.summary(line_length=line_length, print_fn=lambda line: logger.info(line))
    model.rnn_layer.summary(line_length=line_length, print_fn=lambda line: logger.info(line))
    model.summary(line_length=line_length, print_fn=lambda line: logger.info(line))

    logger.info('image_size: {}, model output sizes: {}, max_word_batch: {}, crop_size: {}'.format(image_size, [s.numpy() for s in model.output_sizes], FLAGS.max_word_batch, list(model.crop_size)))

    def create_dataset_from_tfrecord(name, dataset_dirs, is_training):
        def scan_dirs(dirname):
            filenames = []
            for fn in os.listdir(dirname):
                fn = os.path.join(dirname, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)
                elif os.path.isdir(fn) or os.path.islink(fn):
                    filenames += scan_dirs(fn)

            return filenames

        filenames = []
        for dirname in dataset_dirs:
            filenames += scan_dirs(dirname)

        np.random.shuffle(filenames)

        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
        ds = ds.map(lambda record: unpack_tfrecord(record, anchors_all,
                        image_size, FLAGS.max_sequence_len, dict_table, pad_value,
                        is_training, FLAGS.data_format, dtype),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def filter_fn(filename, image, true_values):
            true_word_obj = true_values[..., 0]
            true_word_poly = true_values[..., 1 : 9]
            true_words = true_values[..., 9 : 9 + FLAGS.max_sequence_len]
            true_lengths = true_values[..., 9 + FLAGS.max_sequence_len]
            true_lengths = tf.cast(true_lengths, tf.int64)

            index = tf.logical_and(tf.math.not_equal(true_lengths, 0),
                                   tf.math.not_equal(true_word_obj, 0))
            index = tf.cast(index, tf.int32)
            index_sum = tf.reduce_sum(index)

            return tf.math.not_equal(index_sum, 0)

        ds = ds.filter(filter_fn)

        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(FLAGS.batch_size)

        logger.info('{} object detection dataset has been created, tfrecords: {}'.format(name, len(filenames)))

        return ds

    if FLAGS.dataset_type == 'tfrecords':
        train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, is_training=True)
        train_warmup_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir_warmup, is_training=True)
        eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, is_training=False)

    if FLAGS.save_examples > 0:
        draw_bboxes(image_size, train_dataset, FLAGS.save_examples, anchors_all, FLAGS.dictionary, pad_value, FLAGS.max_sequence_len)
        exit(0)

    logger.info('steps_per_train_epoch: {}, steps_per_warmup_epoch: {}, steps_per_eval_epoch: {}, dictionary_size: {}'.format(
        FLAGS.steps_per_train_epoch, FLAGS.steps_per_warmup_epoch, FLAGS.steps_per_eval_epoch, dictionary_size))

    opt = tfa.optimizers.RectifiedAdam(lr=learning_rate, min_lr=FLAGS.min_learning_rate)
    opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
    if FLAGS.use_fp16:
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

    epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    restore_path = None

    if FLAGS.use_good_checkpoint:
        restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
        if restore_path:
            status = checkpoint.restore(restore_path)
            logger.info("Restored from good checkpoint {}, global step: {}".format(restore_path, global_step.numpy()))

    if not restore_path:
        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}, epoch: {}".format(manager.latest_checkpoint, global_step.numpy(), epoch_var.numpy()))
            restore_path = manager.latest_checkpoint
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

            if FLAGS.base_checkpoint:
                base_checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model.body)
                status = base_checkpoint.restore(FLAGS.base_checkpoint)
                status.expect_partial()

                saved_path = manager.save()
                logger.info("Restored base model from external checkpoint {} and saved object-based checkpoint {}".format(FLAGS.base_checkpoint, saved_path))
                exit(0)

    if FLAGS.use_predicted_polys_epochs >= 0:
        if epoch_var.numpy() >= FLAGS.use_predicted_polys_epochs:
            logger.info('epoch: {}, global_step: {}, use_predicted_polys_epochs: {}, will use predicted polygones for dimensions'.format(
                int(epoch_var.numpy()), global_step.numpy(), FLAGS.use_predicted_polys_epochs))

    if FLAGS.force_epoch:
        logger.info('epoch: {}, global_step: {}, forcing epoch number {}'.format(int(epoch_var.numpy()), global_step.numpy(), FLAGS.force_epoch))
        epoch_var.assign(FLAGS.force_epoch)

    metric = loss.LossMetricAggregator(FLAGS.max_sequence_len, dictionary_size, FLAGS.batch_size)

    anchors_all_batched = tf.expand_dims(anchors_all, 0)
    anchors_all_batched = tf.tile(anchors_all_batched, [FLAGS.batch_size, 1, 1])

    def calculate_metrics(images, is_training, true_values):
        true_word_obj = true_values[..., 0]
        true_word_poly = true_values[..., 1 : 9]
        true_words = true_values[..., 9 : 9 + FLAGS.max_sequence_len]
        true_lengths = true_values[..., 9 + FLAGS.max_sequence_len]

        true_word_obj = tf.cast(true_word_obj, tf.bool)
        true_words = tf.cast(true_words, tf.int64)
        true_lengths = tf.cast(true_lengths, tf.int64)
        true_word_poly = tf.cast(true_word_poly, dtype)

        logits, rnn_features = model(images, is_training)

        use_predicted_polys = False
        if FLAGS.use_predicted_polys_epochs >= 0:
            # this is eager mode, there is no epoch_var.numpy() method
            use_predicted_polys = epoch_var > FLAGS.use_predicted_polys_epochs

        num_words = tf.math.count_nonzero(true_word_obj)

        if num_words > FLAGS.max_word_batch:
            batch_size = tf.shape(true_words)[0]
            feature_size = tf.shape(true_word_obj)[1]

            idx = tf.range(feature_size)
            idx = tf.expand_dims(idx, 0)
            idx = tf.tile(idx, [batch_size, 1])

            batch_idx = tf.range(batch_size)
            batch_idx = tf.expand_dims(batch_idx, 1)
            batch_idx = tf.tile(batch_idx, [1, feature_size])

            true_word_obj_mask = tf.cast(true_word_obj, tf.bool)
            idx_masked = tf.boolean_mask(idx, true_word_obj_mask)
            batch_idx_masked = tf.boolean_mask(batch_idx, true_word_obj_mask)

            ratio = float(FLAGS.max_word_batch) / tf.cast(num_words, tf.float32)
            rnd_idx = tf.random.uniform([num_words], minval=0., maxval=1.)
            rnd_idx = tf.where(rnd_idx < ratio)
            rnd_idx = rnd_idx[:FLAGS.max_word_batch, ...]
            num_updates = tf.shape(rnd_idx)[0]

            scatter_idx = tf.gather(idx_masked, rnd_idx)
            scatter_batch_idx = tf.gather(batch_idx_masked, rnd_idx)

            ones = tf.ones(num_updates, tf.bool)

            scatter_idx_concat = tf.concat([scatter_batch_idx, scatter_idx], 1)

            tf.print('true_words reduced:', num_words, '->', num_updates)

            true_word_obj = tf.scatter_nd(scatter_idx_concat, ones, [batch_size, feature_size])

        rnn_outputs, rnn_outputs_ar = model.rnn_inference_from_true_values(logits, rnn_features,
                                                                           true_word_obj, true_word_poly, true_words, true_lengths,
                                                                           anchors_all, is_training, use_predicted_polys)

        text_loss = metric.text_recognition_loss(true_word_obj, true_words, true_lengths, rnn_outputs, rnn_outputs_ar, is_training)
        objdet_loss = metric.object_detection_loss(true_word_obj, true_word_poly, logits, is_training)

        objdet_scale = 1e-3
        if epoch_var > FLAGS.warmup_objdet_epochs:
            objdet_scale = 1.

        return objdet_loss*objdet_scale, text_loss

    @tf.function
    def eval_step(filenames, images, true_values):
        objdet_loss, text_loss = calculate_metrics(images, False, true_values)
        metric.eval_metric.total_loss.update_state(text_loss)
        return objdet_loss, text_loss

    @tf.function
    def train_step(filenames, images, true_values):
        with tf.GradientTape(persistent=True) as tape:
            objdet_loss, text_loss = calculate_metrics(images, True, true_values)
            if FLAGS.use_fp16:
                scaled_objdet_loss = opt.get_scaled_loss(objdet_loss)
                scaled_text_loss = opt.get_scaled_loss(text_loss)

        tape = hvd.DistributedGradientTape(tape)

        variables = model.trainable_variables
        if FLAGS.use_fp16:
            scaled_text_gradients = tape.gradient(scaled_text_loss, variables)
            scaled_objdet_gradients = tape.gradient(scaled_objdet_loss, variables)

            text_gradients = opt.get_unscaled_gradients(scaled_text_gradients)
            objdet_gradients = opt.get_unscaled_gradients(scaled_objdet_gradients)
        else:
            text_gradients = tape.gradient(text_loss, variables)
            objdet_gradients = tape.gradient(objdet_loss, variables)

        del tape

        stddev = 1 / ((1 + epoch_var)**0.55)

        def return_gradients(prefix, gradients, variables):
            ret_gradients = []
            ret_vars = []
            for g, v in zip(gradients, variables):
                if g is None:
                    #logger.info('no gradients for variable: {}'.format(v))
                    continue

                if g.dtype == tf.float16:
                    logger.info('grad: {}, var: {}'.format(g, v))

                if hvd.rank() == 0 and tf.math.floormod(global_step, 20) == 0:
                    if 'kernel' in v.name:
                        tf.summary.scalar('{}_variables/{}'.format(prefix, v.name), tf.reduce_mean(tf.abs(v)), step=global_step)
                        tf.summary.scalar('{}_gradients/{}'.format(prefix, v.name), tf.reduce_mean(tf.abs(g)), step=global_step)
                        #tf.summary.histogram('{}/gradients/{}'.format(prefix, v.name), g, step=global_step)

                #if epoch_var < 10:
                #    g += tf.random.normal(stddev=stddev, mean=0., shape=g.shape)

                ret_gradients.append(g)
                ret_vars.append(v)

            return ret_gradients, ret_vars

        text_gradients, text_vars = return_gradients('text', text_gradients, variables)
        objdet_gradients, objdet_vars = return_gradients('objdet', objdet_gradients, variables)

        opt.apply_gradients(zip(text_gradients, text_vars))
        opt.apply_gradients(zip(objdet_gradients, objdet_vars))

        global_step.assign_add(1)

        metric.train_metric.total_loss.update_state(text_loss)
        return objdet_loss, text_loss

    def run_epoch(name, dataset, step_func, max_steps, broadcast_variables=False):
        if name == 'train':
            m = metric.train_metric
        else:
            m = metric.eval_metric

        step = 0
        def log_progress():
            if name == 'train':
                logger.info('{}: step: {} {}/{}: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), step, max_steps,
                    metric.str_result(True),
                    ))

        first_batch = True
        for filenames, images, true_values in dataset:
            objdet_loss, text_loss = step_func(filenames, images, true_values)

            if name == 'train':
                if first_batch and broadcast_variables:
                    logger.info('broadcasting initial variables')
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(opt.variables(), root_rank=0)
                    first_batch = False

                if (step % FLAGS.print_per_train_steps == 0) or np.isnan(objdet_loss.numpy()) or np.isnan(text_loss.numpy()):
                    log_progress()

                    if hvd.rank() == 0:
                        tf.summary.scalar('{}/text_loss'.format(name), text_loss, step=global_step)
                        tf.summary.scalar('{}/objdet_loss'.format(name), objdet_loss, step=global_step)

                        tf.summary.scalar('{}/text_ar_acc3'.format(name), m.text_metric_ar.word3_acc.result(), step=global_step)
                        tf.summary.scalar('{}/text_ar_acc'.format(name), m.text_metric_ar.word_acc.result(), step=global_step)
                        tf.summary.scalar('{}/text_ar_acc_full'.format(name), m.text_metric_ar.full_acc.result(), step=global_step)

                        tf.summary.scalar('{}/word_obj_acc_02'.format(name), m.word_obj_accuracy02.result(), step=global_step)
                        tf.summary.scalar('{}/word_obj_acc_05'.format(name), m.word_obj_accuracy05.result(), step=global_step)
                        tf.summary.scalar('{}/word_obj_whole_acc_05'.format(name), m.word_obj_whole_accuracy05.result(), step=global_step)

                        tf.summary.scalar('{}/word_dist_loss'.format(name), m.word_dist_loss.result(), step=global_step)

                    if np.isnan(objdet_loss.numpy()) or np.isnan(text_loss.numpy()):
                        exit(-1)


            step += 1
            if step >= max_steps:
                break

        if hvd.rank() == 0:
            tf.summary.scalar('{}/text_ar_acc3'.format(name), m.text_metric_ar.word3_acc.result(), step=global_step)
            tf.summary.scalar('{}/text_ar_acc'.format(name), m.text_metric_ar.word_acc.result(), step=global_step)
            tf.summary.scalar('{}/text_ar_acc_full'.format(name), m.text_metric_ar.full_acc.result(), step=global_step)

            tf.summary.scalar('{}/word_obj_acc_02'.format(name), m.word_obj_accuracy02.result(), step=global_step)
            tf.summary.scalar('{}/word_obj_acc_05'.format(name), m.word_obj_accuracy05.result(), step=global_step)
            tf.summary.scalar('{}/word_obj_whole_acc_05'.format(name), m.word_obj_whole_accuracy05.result(), step=global_step)

            tf.summary.scalar('{}/word_dist_loss'.format(name), m.word_dist_loss.result(), step=global_step)

        log_progress()

        return step

    best_metric = 0
    best_saved_path = None
    num_epochs_without_improvement = 0
    initial_learning_rate_multiplier = 0.2
    learning_rate_multiplier = initial_learning_rate_multiplier

    def validation_metric():
        return metric.evaluation_result()

    if hvd.rank() == 0:
        if restore_path:
            metric.reset_states()
            logger.info('there is a checkpoint {}, running initial validation'.format(restore_path))

            eval_steps = run_epoch('eval', eval_dataset, eval_step, FLAGS.steps_per_eval_epoch, broadcast_variables=False)
            best_metric = validation_metric()
            logger.info('initial validation: {}, metric: {:.3f}'.format(metric.str_result(False), best_metric))

        if best_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {:.3f} -> {} from command line arguments'.format(best_metric, FLAGS.min_eval_metric))
            best_metric = FLAGS.min_eval_metric

    num_vars = len(model.trainable_variables)
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

    logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, model trainable variables/params: {}/{}'.format(
        num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
        num_vars, int(num_params)))

    learning_rate.assign(FLAGS.initial_learning_rate)
    for epoch in range(FLAGS.num_epochs):
        metric.reset_states()
        want_reset = False

        if epoch_var.numpy() < FLAGS.warmup_tfrecrods_epochs:
            train_steps = run_epoch('train', train_warmup_dataset, train_step, FLAGS.steps_per_warmup_epoch, (epoch == 0))
        else:
            if epoch_var.numpy() == FLAGS.warmup_tfrecrods_epochs and epoch != 0:
                logger.info('removing warmup data from datasets: lr: {} -> {}'.format(learning_rate.numpy(), FLAGS.initial_learning_rate))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier
                learning_rate.assign(FLAGS.initial_learning_rate)
                want_reset = True

            train_steps = run_epoch('train', train_dataset, train_step, FLAGS.steps_per_train_epoch, (epoch == 0))

        eval_steps = run_epoch('eval', eval_dataset, eval_step, FLAGS.steps_per_eval_epoch, broadcast_variables=False)

        epoch_var.assign_add(1)

        new_lr = learning_rate.numpy()

        new_metric = validation_metric()

        logger.info('epoch: {}, train: steps: {}, lr: {:.2e}, train: {}, eval: {}, val_metric: {:.4f}/{:.4f}'.format(
            int(epoch_var.numpy()), global_step.numpy(),
            learning_rate.numpy(),
            metric.str_result(True), metric.str_result(False),
            new_metric, best_metric))

        if hvd.rank() == 0:
            saved_path = manager.save()

        if new_metric > best_metric:
            if epoch_var.numpy() > FLAGS.skip_saving_epochs:
                if hvd.rank() == 0:
                    best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, new_metric))

                logger.info("epoch: {}, global_step: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}: {}".format(
                    int(epoch_var.numpy()), global_step.numpy(), best_saved_path, best_metric, new_metric, metric.str_result(False)))

                best_metric = new_metric

            num_epochs_without_improvement = 0
            learning_rate_multiplier = initial_learning_rate_multiplier
        else:
            num_epochs_without_improvement += 1


        if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
            if learning_rate > FLAGS.min_learning_rate:
                new_lr = learning_rate.numpy() * learning_rate_multiplier
                if new_lr < FLAGS.min_learning_rate:
                    new_lr = FLAGS.min_learning_rate

                if FLAGS.reset_on_lr_update:
                    want_reset = True

                logger.info('epoch: {}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))
                num_epochs_without_improvement = 0
                if learning_rate_multiplier > 0.1:
                    learning_rate_multiplier /= 2


            elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                new_lr = FLAGS.initial_learning_rate
                want_reset = True

                logger.info('epoch: {}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

        if FLAGS.warmup_objdet_epochs >= 0:
            if epoch_var.numpy() == FLAGS.warmup_objdet_epochs:
                logger.info('epoch: {}, global_step: {}, starting to use normal objdet scale (1.0) instead of warmup diminished one'.format(
                    int(epoch_var.numpy()), global_step.numpy()))

        if FLAGS.use_predicted_polys_epochs >= 0:
            if epoch_var.numpy() == FLAGS.use_predicted_polys_epochs:
                logger.info('epoch: {}, global_step: {}, starting to use predicted polygones for dimensions, will reset model to the best available'.format(
                    int(epoch_var.numpy()), global_step.numpy()))

                want_reset = True

        if want_reset:
            restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
            if restore_path:
                epoch_num = epoch_var.numpy()
                step_num = global_step.numpy()
                logger.info('epoch: {}, global_step: {}, best metric: {:.5f}, learning rate: {:.2e} -> {:.2e}, restoring best checkpoint: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), best_metric, learning_rate.numpy(), new_lr, best_saved_path))

                checkpoint.restore(best_saved_path)

                epoch_var.assign(epoch_num)
                global_step.assign(step_num)

        # update learning rate even without resetting model
        learning_rate.assign(new_lr)


if __name__ == '__main__':
    hvd.init()

    random_seed = int.from_bytes(os.urandom(4), 'big')

    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    if hvd.rank() == 0:
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
