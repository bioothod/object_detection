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
parser.add_argument('--train_tfrecord_dir', type=str, action='append', help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, action='append', help='Directory containing evaluation TFRecords')
parser.add_argument('--image_size', type=int, required=True, help='Use this image size, if 0 - use default')
parser.add_argument('--steps_per_eval_epoch', default=30, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_train_epoch', default=200, type=int, help='Number of steps per train run')
parser.add_argument('--save_examples', type=int, default=0, help='Number of example images to save and exit')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--disable_rotation_augmentation', action='store_true', help='Whether to disable rotation/flipping augmentation')

default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
parser.add_argument('--dictionary', type=str, default=default_char_dictionary, help='Dictionary to use')

def create_lookup_table(dictionary):
    dict_split = tf.strings.unicode_split(dictionary, 'UTF-8')
    dictionary_size = dict_split.shape[0] + 1
    kv_init = tf.lookup.KeyValueTensorInitializer(keys=dict_split, values=tf.range(1, dictionary_size, 1), key_dtype=tf.string, value_dtype=tf.int32)
    dict_table = tf.lookup.StaticHashTable(kv_init, 0)

    return dictionary_size, dict_table

def unpack_tfrecord(record, anchors_all, image_size, dictionary_size, dict_table, is_training, data_format):
    features = tf.io.parse_single_example(record,
            features={
                'filename': tf.io.FixedLenFeature([], tf.string),
                'word_poly': tf.io.FixedLenFeature([], tf.string),
                'char_poly': tf.io.FixedLenFeature([], tf.string),
                'text': tf.io.FixedLenFeature([], tf.string),
                'text_concat': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
            })

    filename = features['filename']

    image = features['image']
    image = tf.image.decode_jpeg(image, channels=3)

    char_poly = features['char_poly']
    char_poly = tf.io.decode_raw(char_poly, out_type=tf.float64)
    char_poly = tf.cast(char_poly, tf.float32)
    char_poly = tf.reshape(char_poly, [-1, 4, 2])

    word_poly = features['word_poly']
    word_poly = tf.io.decode_raw(word_poly, out_type=tf.float32)
    word_poly = tf.reshape(word_poly, [-1, 4, 2])

    text = features['text_concat']

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2, tf.int32),
                tf.cast((mx - orig_image_width) / 2, tf.int32),
                mx_int,
                mx_int)

    xdiff = (mx - orig_image_width) / 2
    ydiff = (mx - orig_image_height) / 2

    add = tf.stack([xdiff, ydiff])
    word_poly += add
    char_poly += add

    image = tf.image.resize(image, [image_size, image_size])

    scale = image_size / mx
    word_poly *= scale
    char_poly *= scale

    image = tf.cast(image, tf.float32)
    image -= 128
    image /= 128

    chars = tf.strings.unicode_split(text, 'UTF-8')
    encoded_chars = dict_table.lookup(chars)

    if is_training:
        image, char_poly, word_poly = preprocess.preprocess_for_train(image, char_poly, word_poly, image_size, FLAGS.disable_rotation_augmentation)

    true_values = anchors_gen.generate_true_values_for_anchors(char_poly, word_poly, encoded_chars, anchors_all, dictionary_size)

    return filename, image, true_values

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def draw_bboxes(image_size, train_dataset, num_examples, all_anchors, dictionary, dictionary_size):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    for filename, image, true_values in train_dataset.unbatch().take(num_examples):
        filename = str(filename.numpy(), 'utf8')
        filename_base = os.path.basename(filename)

        dst = os.path.join(data_dir, filename_base)

        char_boundary_start = 0
        word_boundary_start = dictionary_size + 1 + 4 * 2

        y_true = true_values.numpy()
        logger.info(filename)

        # true tensors
        true_char = y_true[..., char_boundary_start : char_boundary_start + word_boundary_start]
        true_word = y_true[..., word_boundary_start : ]

        true_char_obj = true_char[..., 0]
        true_char_poly = true_char[..., 1 : 9]
        true_char_letters = true_char[..., 10 :]

        true_word_obj = true_word[..., 0]
        true_word_poly = true_word[..., 1 : 9]

        char_index = np.where(true_char_obj != 0)[0]
        word_index = np.where(true_word_obj != 0)[0]

        char_poly = tf.gather(true_char_poly, char_index)
        char_poly = tf.reshape(char_poly, [-1, 4, 2])
        word_poly = tf.gather(true_word_poly, word_index).numpy()
        word_poly = tf.reshape(word_poly, [-1, 4, 2])

        best_anchors = tf.gather(all_anchors[..., :2], char_index)
        best_anchors = tf.expand_dims(best_anchors, 1)
        best_anchors = tf.tile(best_anchors, [1, 4, 1])
        char_poly = char_poly + best_anchors

        best_anchors = tf.gather(all_anchors[..., :2], word_index)
        best_anchors = tf.expand_dims(best_anchors, 1)
        best_anchors = tf.tile(best_anchors, [1, 4, 1])
        word_poly = word_poly + best_anchors

        image_filename = '/shared2/object_detection/datasets/text/synth_text/SynthText/{}'.format(filename)
        if False:
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = image.numpy()
            image = image * 128 + 128
            image = image.astype(np.uint8)

        imh, imw = image.shape[:2]
        max_side = max(imh, imw)
        pad_y = (max_side - imh) / 2
        pad_x = (max_side - imw) / 2
        square_scale = max_side / image_size

        char_poly = char_poly.numpy()
        word_poly = word_poly.numpy()

        char_poly *= square_scale
        word_poly *= square_scale


        diff = [pad_x, pad_y]
        char_poly -= diff
        word_poly -= diff


        new_anns = []
        for poly in word_poly:
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

        dictionary_size, dict_table = create_lookup_table(FLAGS.dictionary)

        image_size = FLAGS.image_size
        #num_classes = 1 + 4*2 + dictionary_size + 1 + 4*2
        model = encoder.create_model(FLAGS.model_name, dictionary_size)
        if model.output_sizes is None:
            dummy_input = tf.ones((int(FLAGS.batch_size / num_replicas), image_size, image_size, 3), dtype=dtype)
            dstrategy.experimental_run_v2(lambda m, inp: m(inp, True), args=(model, dummy_input))
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

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
            ds = ds.map(lambda record: unpack_tfrecord(record, anchors_all,
                            image_size, dictionary_size, dict_table,
                            is_training, FLAGS.data_format),
                    num_parallel_calls=FLAGS.num_cpus)

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
            draw_bboxes(image_size, train_dataset, FLAGS.save_examples, anchors_all, FLAGS.dictionary, dictionary_size)
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

        metric = loss.LossMetricAggregator(dictionary_size, output_xy_grids, FLAGS.batch_size)

        def calculate_metrics(images, is_training, true_values):
            logits = model(images, training=is_training)
            total_loss = metric.loss(true_values, logits, is_training)
            return total_loss

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

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

                if want_reset and best_saved_path is not None:
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
