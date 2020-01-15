import argparse
import cv2
import logging
import os
import sys

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')


import anchors_gen
import encoder
import image as image_draw
import loss
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
parser.add_argument('--print_per_train_steps', default=100, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--negative_positive_rate', default=2, type=float, help='Negative to positive anchors ratio')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--dataset_type', type=str, choices=['tfrecords'], default='tfrecords', help='Dataset type')
parser.add_argument('--train_tfrecord_dir', type=str, required=True, action='append', help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, required=True, action='append', help='Directory containing evaluation TFRecords')
parser.add_argument('--image_size', type=int, required=True, help='Use this image size, if 0 - use default')
parser.add_argument('--steps_per_eval_epoch', default=30, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_train_epoch', default=200, type=int, help='Number of steps per train run')
parser.add_argument('--save_examples', type=int, default=0, help='Number of example images to save and exit')
parser.add_argument('--max_sequence_len', type=int, default=32, help='Maximum word length, also number of RNN attention timesteps')
parser.add_argument('--min_sequence_len', type=int, default=32, help='Minimum number of time steps in training')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--disable_rotation_augmentation', action='store_true', help='Whether to disable rotation/flipping augmentation')

default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
parser.add_argument('--dictionary', type=str, default=default_char_dictionary, help='Dictionary to use')

def unpack_tfrecord(record, anchors_all, image_size, max_sequence_len, dict_table, pad_value, is_training, data_format):
    features = tf.io.parse_single_example(record,
            features={
                'filename': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
                'true_labels': tf.io.FixedLenFeature([], tf.string),
                'true_bboxes': tf.io.FixedLenFeature([], tf.string),
            })

    filename = features['filename']

    image = features['image']
    image = tf.image.decode_jpeg(image, channels=3)

    orig_bboxes = tf.io.decode_raw(features['true_bboxes'], tf.float32)
    orig_bboxes = tf.reshape(orig_bboxes, [-1, 4])
    cx, cy, h, w = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)
    xmin = cx - w / 2
    ymin = cy - h / 2

    # return bboxes in the original format without scaling it to square
    p0 = tf.concat([xmin, ymin], axis=1)
    p1 = tf.concat([xmin + w, ymin], axis=1)
    p2 = tf.concat([xmin + w, ymin + h], axis=1)
    p3 = tf.concat([xmin, ymin + h], axis=1)

    word_poly = tf.stack([p0, p1, p2, p3], axis=1)

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

    image = tf.cast(image, tf.float32)
    image -= 128
    image /= 128


    if is_training:
        image, word_poly, reverse_text_labels = preprocess.preprocess_for_train(image, word_poly, image_size, FLAGS.disable_rotation_augmentation)

    text_labels = tf.strings.split(features['true_labels'], '<SEP>')
    text_split = tf.strings.unicode_split(text_labels, 'UTF-8')
    if reverse_text_labels:
        text_split = tf.reverse(text_split, [1])

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

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def draw_bboxes(image_size, train_dataset, num_examples, all_anchors, dictionary, max_sequence_len):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    for filename, image, true_values in train_dataset.unbatch().take(num_examples):
        filename = str(filename.numpy(), 'utf8')
        filename_base = os.path.basename(filename)

        dst = os.path.join(data_dir, filename_base)

        image_filename = '/shared2/object_detection/datasets/text/synth_text/SynthText/{}'.format(filename)
        if False:
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = image.numpy()
            image = image * 128 + 128
            image = image.astype(np.uint8)

        true_word_obj, word_poly, text_labels, lengths = anchors_gen.unpack_true_values(true_values, all_anchors, image.shape, image_size, max_sequence_len)

        word_poly = word_poly.numpy()
        text_labels = text_labels.numpy()

        new_anns = []
        for poly, text in zip(word_poly, text_labels):
            new_anns.append((None, poly, None))

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

        dictionary_size, dict_table, pad_value = anchors_gen.create_lookup_table(FLAGS.dictionary)

        image_size = FLAGS.image_size
        model = encoder.create_model(FLAGS.model_name, FLAGS.max_sequence_len, dictionary_size, pad_value)
        if model.output_sizes is None:
            dummy_input = tf.ones((int(FLAGS.batch_size / num_replicas), image_size, image_size, 3), dtype=dtype)

            dstrategy.experimental_run_v2(
                    lambda m, inp: m(inp, training=True),
                    args=(model, dummy_input))

            logger.info('image_size: {}, model output sizes: {}'.format(image_size, model.output_sizes))

        anchors_all, output_xy_grids, output_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

        def create_dataset_from_tfrecord(name, dataset_dirs, is_training):
            filenames = []
            for dirname in dataset_dirs:
                for fn in os.listdir(dirname):
                    fn = os.path.join(dirname, fn)
                    if os.path.isfile(fn):
                        filenames.append(fn)

            np.random.shuffle(filenames)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=FLAGS.num_cpus)
            ds = ds.map(lambda record: unpack_tfrecord(record, anchors_all,
                            image_size, FLAGS.max_sequence_len, dict_table, pad_value,
                            is_training, FLAGS.data_format),
                    num_parallel_calls=FLAGS.num_cpus)

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

            ds = ds.batch(FLAGS.batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
            if FLAGS.save_examples <= 0:
                ds = dstrategy.experimental_distribute_dataset(ds)

            logger.info('{} dataset has been created, tfrecords: {}'.format(name, len(filenames)))

            return ds

        if FLAGS.dataset_type == 'tfrecords':
            train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, is_training=True)
            eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, is_training=False)

        if FLAGS.save_examples > 0:
            draw_bboxes(image_size, train_dataset, FLAGS.save_examples, anchors_all, FLAGS.dictionary, FLAGS.max_sequence_len)
            exit(0)

        steps_per_train_epoch = FLAGS.steps_per_train_epoch
        steps_per_eval_epoch = FLAGS.steps_per_eval_epoch

        logger.info('steps_per_train_epoch: {}, steps_per_eval_epoch: {}, dictionary_size: {}'.format(
            steps_per_train_epoch, steps_per_eval_epoch, dictionary_size))

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

        metric = loss.LossMetricAggregator(FLAGS.max_sequence_len, dictionary_size, FLAGS.batch_size)

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        current_max_sequence_len = tf.Variable(FLAGS.min_sequence_len, dtype=tf.int32, name='current_max_sequence_len', trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)


        def calculate_metrics(images, is_training, true_values):
            true_word_obj = true_values[..., 0]
            true_word_poly = true_values[..., 1 : 9]
            true_words = true_values[..., 9 : 9 + FLAGS.max_sequence_len]
            true_lengths = true_values[..., 9 + FLAGS.max_sequence_len]
            true_words = tf.cast(true_words, tf.int64)
            true_lengths = tf.cast(true_lengths, tf.int64)

            logits, rnn_features = model(images, is_training)
            rnn_logits = model.rnn_inference_from_true_values(rnn_features, true_word_obj, true_word_poly, true_words, true_lengths, anchors_all, is_training)

            total_loss = metric.loss(true_values, logits, rnn_logits, current_max_sequence_len, is_training)
            return total_loss

        def eval_step(filenames, images, true_values):
            total_loss = calculate_metrics(images, False, true_values)
            return total_loss

        def train_step(filenames, images, true_values):
            with tf.GradientTape() as tape:
                total_loss = calculate_metrics(images, True, true_values)

            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)

            stddev = 1 / ((1 + epoch_var)**0.55)

            clip_gradients = []
            for g, v in zip(gradients, variables):
                if g is None:
                    pass
                    #logger.info('no gradients for variable: {}'.format(v))
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
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return total_loss

        @tf.function
        def distributed_eval_step(args):
            total_loss = dstrategy.experimental_run_v2(eval_step, args=args)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
            return total_loss

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            def log_progress():
                if name == 'train':
                    logger.info('{}: {}: step: {}/{}, {}'.format(
                        name, int(epoch_var.numpy()), step, max_steps,
                        metric.str_result(True),
                        ))

            for filenames, images, true_values in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                total_loss = step_func(args=(filenames, images, true_values))
                if (name == 'train' and step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                    log_progress()

                    if np.isnan(total_loss.numpy()):
                        exit(-1)


                step += 1
                if step >= max_steps:
                    break

            log_progress()

            return step

        best_metric = 0
        best_saved_path = None
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier

        def validation_metric():
            return metric.evaluation_result()

        if manager.latest_checkpoint:
            metric.reset_states()
            logger.info('there is a checkpoint {}, running initial validation'.format(manager.latest_checkpoint))

            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval_epoch)
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
        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            epoch_var.assign(epoch)

            metric.reset_states()

            train_steps = run_epoch('train', train_dataset, distributed_train_step, steps_per_train_epoch)
            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval_epoch)

            new_metric = validation_metric()

            logger.info('epoch: {}, train: steps: {}, lr: {:.2e}, train: {}, eval: {}, val_metric: {:.4f}/{:.4f}'.format(
                epoch, global_step.numpy(),
                learning_rate.numpy(),
                metric.str_result(True), metric.str_result(False),
                new_metric, best_metric))

            saved_path = manager.save()

            if new_metric > best_metric:
                best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, new_metric))

                logger.info("epoch: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}: {}".format(
                    epoch, best_saved_path, best_metric, new_metric, metric.str_result(False)))
                best_metric = new_metric
                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                want_reset = False
                new_lr = learning_rate.numpy()

                if learning_rate > FLAGS.min_learning_rate:
                    new_lr = learning_rate.numpy() * learning_rate_multiplier
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr))
                    num_epochs_without_improvement = 0
                    if learning_rate_multiplier > 0.1:
                        learning_rate_multiplier /= 2

                    if FLAGS.reset_on_lr_update:
                        want_reset = True
                elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                    new_lr = FLAGS.initial_learning_rate
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr))
                    num_epochs_without_improvement = 0
                    want_reset = True
                    learning_rate_multiplier = initial_learning_rate_multiplier

                if want_reset and best_saved_path:
                    logger.info('epoch: {}, best metric: {:.5f}, learning rate: {:.2e} -> {:.2e}, restoring best checkpoint: {}'.format(
                        epoch, best_metric, learning_rate.numpy(), new_lr, best_saved_path))

                    checkpoint.restore(best_saved_path)

                learning_rate.assign(new_lr)


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
