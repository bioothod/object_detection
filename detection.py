import argparse
import cv2
import json
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa

import horovod.tensorflow as hvd

logger = logging.getLogger('detection')

import anchors
import autoaugment
import callbacks
import config
import encoder
import evaluate
import image as image_draw
import metric
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd', 'radam_lookahead', 'modern'], help='Optimizer')
parser.add_argument('--category_json', type=str, required=True, help='Category to ID mapping json file.')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
parser.add_argument('--eval_batch_size', type=int, default=128, help='Number of images to process in a batch.')
parser.add_argument('--max_items_in_image', type=int, default=32, help='Limit number of true objects in image.')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run.')
parser.add_argument('--epoch', type=int, default=0, help='Initial epoch\'s number')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs.')
parser.add_argument('--clip_grad', type=float, default=10, help='Clip accumulated gradients by this value')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored.')
parser.add_argument('--base_checkpoint', type=str, help='Load base model weights from this file')
parser.add_argument('--use_good_checkpoint', action='store_true', help='Recover from the last good checkpoint when present')
parser.add_argument('--d', type=int, default=0, help='Model name suffix: 0-7')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
parser.add_argument('--steps_per_eval_epoch', default=-1, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_train_epoch', default=-1, type=int, help='Number of steps per training run')
parser.add_argument('--print_per_train_steps', default=100, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--epochs_lr_update', default=10, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--lr_scheduler', type=str, choices=['reset', 'cosine'], default='reset', help='Learning rate scheduler')
parser.add_argument('--lr_warmup_steps', default=0, type=int, help='Learning rate warmup steps')
parser.add_argument('--lr_decay_epochs', default=0, type=int, help='Learning rate decays epochs')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--dataset_type', type=str, choices=['tfrecords', 'annotations'], default='tfrecords', help='Dataset type')
parser.add_argument('--train_tfrecord_pattern', type=str, help='Training TFRecords pattern')
parser.add_argument('--eval_tfrecord_pattern', type=str, help='Evaluation TFRecords pattern')
parser.add_argument('--train_annotations_json', type=str, help='Training annotations pattern')
parser.add_argument('--eval_annotations_json', type=str, help='Evaluation annotations pattern')
parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset')
parser.add_argument('--train_num_images', type=int, default=-1, help='Number of images in train epoch')
parser.add_argument('--eval_num_images', type=int, default=-1, help='Number of images in eval epoch')
parser.add_argument('--grad_accumulate_steps', type=int, default=1, help='Number of batches to accumulate before gradient update')
parser.add_argument('--reg_loss_weight', type=float, default=0, help='L2 regularization weight')
parser.add_argument('--only_test', action='store_true', help='Exist after running initial validation')
parser.add_argument('--run_evaluation_first', action='store_true', help='Run evaluation before the first training epoch')
parser.add_argument('--train_echo_factor', type=int, default=1, help='Repeat augmented examples this many times in shuffle buffer before batching and training')
parser.add_argument('--class_activation', type=str, default='softmax', help='Classification activation function')
parser.add_argument('--rotation_augmentation', type=int, default=-1, help='Angle for rotation augmentation')
parser.add_argument('--use_augmentation', type=str, help='Use efficientnet random/v0/distort augmentation')
parser.add_argument('--save_examples', type=int, default=0, help='Save this many train examples')

def polygon2bbox(poly, want_yx=False, want_wh=False):
    # polygon shape [N, 4, 2]

    x = poly[..., 0]
    y = poly[..., 1]

    xmin = tf.math.reduce_min(x, axis=1, keepdims=True)
    ymin = tf.math.reduce_min(y, axis=1, keepdims=True)
    xmax = tf.math.reduce_max(x, axis=1, keepdims=True)
    ymax = tf.math.reduce_max(y, axis=1, keepdims=True)

    if want_yx:
        bbox = tf.concat([ymin, xmin, ymax, xmax], 1)
    elif want_wh:
        bbox = tf.concat([xmin, ymin, xmax - xmin, ymax - ymin], 1)
    else:
        bbox = tf.concat([xmin, ymin, xmax, ymax], 1)

    return bbox

def prepare_example(filename, image_id, image, orig_bboxes, orig_labels, image_size, num_classes, is_training, data_format):
    x0, y0, w, h = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)

    orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
    orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image,
                tf.cast((mx - orig_image_height) / 2, tf.int32),
                tf.cast((mx - orig_image_width) / 2, tf.int32),
                mx_int,
                mx_int)
    x0 += (mx - orig_image_width) / 2
    y0 += (mx - orig_image_height) / 2

    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    dtype = image.dtype

    p0 = tf.concat([x0, y0], 1)
    p1 = tf.concat([x0+w, y0], 1)
    p2 = tf.concat([x0+w, y0+h], 1)
    p3 = tf.concat([x0, y0+h], 1)

    word_poly = tf.stack([p0, p1, p2, p3], 1)

    if is_training:
        image, word_poly, new_labels = preprocess.preprocess_for_train(image, word_poly, orig_labels, FLAGS.rotation_augmentation, FLAGS.use_augmentation, dtype)
    else:
        image = preprocess.preprocess_for_evaluation(image, dtype)
        new_labels = orig_labels

    current_image_size = tf.cast(tf.shape(image)[1], tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, dtype)

    word_poly = word_poly / current_image_size * image_size
    word_poly = tf.cast(word_poly, dtype)

    new_bboxes = polygon2bbox(word_poly)

    new_bboxes = new_bboxes[:FLAGS.max_items_in_image, ...]
    new_labels = new_labels[:FLAGS.max_items_in_image]

    return filename, image_id, image, new_bboxes, new_labels

def unpack_tfrecord(serialized_example, image_size, num_classes, is_training, data_format, dtype):
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

    orig_labels = tf.io.decode_raw(features['true_labels'], tf.int32)

    image_id = features['image_id']
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.cast(image, dtype)

    return prepare_example(filename, image_id, image, orig_bboxes, orig_labels, image_size, num_classes, is_training, data_format)

def tf_read_image(filename, image_id, bboxes, labels, image_labels, image_size, num_classes, is_training, data_format, dtype):
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype)
    return prepare_example(filename, image_id, image, bboxes, labels, image_size, num_classes, is_training, data_format)

def calc_epoch_steps(num_files, batch_size):
    return (num_files + batch_size - 1) // batch_size

def draw_bboxes(dataset, cat_names, num_examples):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    for filename, image_id, image, true_bboxes, true_labels in dataset.unbatch().take(num_examples):
        filename = str(filename.numpy(), 'utf8')
        filename_base = os.path.basename(filename)
        filename_base = os.path.splitext(filename_base)[0]

        dst = os.path.join(data_dir, filename_base) + '.png'

        image = image.numpy()
        image = image * 128 + 128
        image = image.astype(np.uint8)

        true_bboxes = true_bboxes.numpy()
        true_labels = true_labels.numpy()

        new_anns = []
        for bb, label in zip(true_bboxes, true_labels):
            if label == -1:
                continue

            new_anns.append((bb, None, label))

        image_draw.draw_im(image, new_anns, dst, cat_names)

def generate_anchors(anchors_config: config.AnchorsConfig,
                     im_shape: int) -> tf.Tensor:

    anchors_gen = [anchors.AnchorGenerator(
                            size=anchors_config.sizes[i - 3],
                            aspect_ratios=anchors_config.ratios,
                            stride=anchors_config.strides[i - 3])
                    for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors_all = [g((size, size, 3)) for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors_all, axis=0)

def filter_fn(filename, image_id, image, true_bboxes, true_labels):
    index = tf.math.not_equal(true_labels, -1)
    index = tf.cast(index, tf.int32)
    index_sum = tf.reduce_sum(index)

    return tf.math.logical_and(
            tf.math.not_equal(index_sum, 0),
            tf.math.not_equal(tf.shape(true_labels)[0], 0))

def wrap_dataset(ds, image_size, dtype, is_training):
    ds = ds.filter(filter_fn)

    if is_training:
        batch_size = FLAGS.batch_size
        if FLAGS.train_echo_factor > 1:
            ds = ds.flat_map(lambda *t: tf.data.Dataset.from_tensors(t).repeat(FLAGS.train_echo_factor))
        ds = ds.shuffle(256)
    else:
        batch_size = FLAGS.eval_batch_size

    pad_value = tf.constant(1, dtype=dtype)

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size=batch_size,
            padded_shapes=((), (), (image_size, image_size, 3), (FLAGS.max_items_in_image, 4), (FLAGS.max_items_in_image,)),
            padding_values=('', tf.constant(0, dtype=tf.int64), 0*pad_value, -1*pad_value, -1))

    ds = ds.repeat()

    return ds

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    good_checkpoint_dir = os.path.join(checkpoint_dir, 'good')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(good_checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log.{}'.format(hvd.rank())), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    with open(FLAGS.category_json, 'r') as f:
        class2idx = json.load(f)

    logger.info('start: {}'.format(' '.join(sys.argv)))
    for k, v in FLAGS.__dict__.items():
        logger.info('  --{}={}'.format(k, v))

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

    logdir = os.path.join(FLAGS.train_dir, 'logs')
    writer = tf.summary.create_file_writer(logdir)

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    epoch_var = tf.Variable(0, dtype=tf.int64, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

    model = encoder.create_model(FLAGS.d, FLAGS.num_classes, class_activation=FLAGS.class_activation, dtype=dtype)
    image_size = model.config.input_size

    anchors_all = generate_anchors(model.anchors_config, model.config.input_size)

    if FLAGS.dataset_type == 'tfrecords':
        import glob

        def create_dataset_from_tfrecord(name, dataset_pattern, image_size, num_classes, is_training):
            filenames = []
            for fn in glob.glob(dataset_pattern):
                if os.path.isfile(fn):
                    filenames.append(fn)

            total_filenames = len(filenames)
            if is_training and hvd.size() > 1 and len(filenames) > hvd.size():
                filenames = np.array_split(filenames, hvd.size())[hvd.rank()]
                np.random.seed(int.from_bytes(os.urandom(4), 'big'))
                np.random.shuffle(filenames)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
            ds = ds.map(lambda record: unpack_tfrecord(record,
                            image_size, num_classes, is_training,
                            FLAGS.data_format, dtype),
                    num_parallel_calls=FLAGS.num_cpus)

            ds = wrap_dataset(ds, image_size, dtype, is_training)

            logger.info('{} dataset has been created, tfrecords: {}/{}'.format(name, len(filenames), total_filenames))
            return ds

        train_num_images = FLAGS.train_num_images
        eval_num_images = FLAGS.eval_num_images
        train_num_classes = FLAGS.num_classes

        if train_num_classes is None:
            logger.error('If there is no train_num_classes (tfrecord dataset), you must provide --num_classes')
            exit(-1)

        train_cat_names = {}
        for cname, cid in class2idx.items():
            train_cat_names[cid] = cname

        train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_pattern, image_size, train_num_classes, is_training=True)
        eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_pattern, image_size, train_num_classes, is_training=False)

        steps_per_train_epoch = FLAGS.steps_per_train_epoch
        if FLAGS.train_num_images > 0:
            steps_per_train_epoch = calc_epoch_steps(FLAGS.train_num_images, FLAGS.batch_size)

        steps_per_eval_epoch = FLAGS.steps_per_eval_epoch
        if FLAGS.eval_num_images > 0:
            steps_per_eval_epoch = calc_epoch_steps(FLAGS.eval_num_images, FLAGS.eval_batch_size)

    elif FLAGS.dataset_type == 'annotations':
        import batch

        def gen(bg, num_images, is_training):
            want_full = not is_training
            yield_num_images = 0

            for filename, image_id, anns, image_anns in zip(*bg.get(num=num_images, want_full=want_full)):
                if yield_num_images == num_images:
                    break

                if not os.path.exists(filename):
                    continue

                bboxes = []
                labels = []
                image_labels = []

                for ann in anns:
                    bboxes.append(ann['bbox'])
                    labels.append(ann['category_id'])

                for ann in image_anns:
                    image_labels.append(ann['category_id'])

                if len(labels) == 0 and len(image_labels) == 0:
                    continue

                #if len(bboxes) > FLAGS.max_items_in_image:
                #    logger.info('{}: bboxes: {}, labels: {}, image_labels: {}'.format(filename, len(bboxes), len(labels), len(image_labels)))

                bboxes = np.array(bboxes, dtype=np.float32).reshape([len(bboxes), 4])
                labels = np.array(labels, dtype=np.int32)
                image_labels = np.array(image_labels, dtype=np.int32)

                bboxes = bboxes[:FLAGS.max_items_in_image, ...]
                labels = labels[:FLAGS.max_items_in_image]
                image_labels = image_labels[:FLAGS.max_items_in_image]

                yield_num_images += 1
                yield filename, image_id, bboxes, labels, image_labels

        def create_dataset_from_annotations(name, ann_file, image_size, num_classes, num_images, is_training):
            bg = batch.generator(ann_file)
            if num_classes is None:
                num_classes = bg.num_classes()

            if num_images is None:
                num_images = bg.num_images()

            ds = tf.data.Dataset.from_generator(lambda: gen(bg, num_images, is_training),
                                    output_types=(tf.string, tf.int64, tf.float32, tf.int32, tf.int32),
                                    output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, 4]), tf.TensorShape([None]), tf.TensorShape([None])))
            ds = ds.map(lambda filename, image_id, bboxes, labels, image_labels:
                            tf_read_image(filename, image_id, bboxes, labels, image_labels,
                                image_size, num_classes, is_training, FLAGS.data_format, dtype),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = wrap_dataset(ds, image_size, dtype, is_training)

            logger.info('{} dataset has been created, total images: {}, categories: {}, image_categories: {}'.format(name, bg.num_images(), bg.num_classes(), bg.num_image_classes()))
            return ds, num_classes, num_images

        def calc_num_images(num_images, steps_per_epoch, batch_size):
            if num_images <= 0:
                if steps_per_epoch > 0:
                    num_images = steps_per_epoch * batch_size
                else:
                    num_images = None

            return num_images

        train_num_images = calc_num_images(FLAGS.train_num_images, FLAGS.steps_per_train_epoch, FLAGS.batch_size)
        eval_num_images = calc_num_images(FLAGS.eval_num_images, FLAGS.steps_per_eval_epoch, FLAGS.batch_size)

        train_dataset, train_num_classes, train_num_images = create_dataset_from_annotations('train', FLAGS.train_annotations_json, image_size,
                num_classes=None, num_images=train_num_images, is_training=True)
        eval_dataset, eval_num_classes, eval_num_images = create_dataset_from_annotations('eval', FLAGS.eval_annotations_json, image_size,
                train_num_classes, num_images=eval_num_images, is_training=False)

        steps_per_train_epoch = calc_epoch_steps(train_num_images, FLAGS.batch_size)
        steps_per_eval_epoch = calc_epoch_steps(eval_num_images, FLAGS.eval_batch_size)

    if FLAGS.save_examples > 0:
        draw_bboxes(train_dataset, train_cat_names, FLAGS.save_examples)
        exit(0)


    logger.info('steps_per_train_epoch: {}, train images: {}, steps_per_eval_epoch: {}, eval images: {}'.format(
        steps_per_train_epoch, train_num_images,
        steps_per_eval_epoch, eval_num_images))

    if FLAGS.optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'modern' or FLAGS.optimizer == 'radam_lookahead':
        opt = tfa.optimizers.RectifiedAdam(lr=learning_rate, min_lr=FLAGS.min_learning_rate)
        opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
    else:
        logger.error('Unsupported optimized \'{}\''.format(FLAGS.optimizer))
        exit(-1)

    if FLAGS.use_fp16:
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

    checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model, epoch=epoch_var)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    restore_path = None
    if FLAGS.use_good_checkpoint:
        restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
        if restore_path:
            status = checkpoint.restore(restore_path)
            epoch_var.assign_add(1)
            logger.info("Restored from good checkpoint {}, global step: {}".format(restore_path, global_step.numpy()))

    if not restore_path:
        status = checkpoint.restore(manager.latest_checkpoint)
        epoch_var.assign_add(1)

        if manager.latest_checkpoint:
            logger.info("Restored from {}, global step: {}, epoch: {}".format(manager.latest_checkpoint, global_step.numpy(), epoch_var.numpy()))
            restore_path = manager.latest_checkpoint
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

        if FLAGS.base_checkpoint:
            base_checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model.body)
            status = base_checkpoint.restore(FLAGS.base_checkpoint)
            status.expect_partial()

            if hvd.rank() == 0:
                saved_path = manager.save()
                logger.info("Restored base model from external checkpoint {} and saved object-based checkpoint {}".format(FLAGS.base_checkpoint, saved_path))
            else:
                logger.info('Exiting because only hvd0 is saving the converted base checkpoint')

            exit(0)

    met = metric.ModelMetric(anchors_all, train_num_classes)

    @tf.function(experimental_relax_shapes=True)
    def train_step(filenames, images, true_bboxes, true_labels):
        with tf.GradientTape() as tape:
            bboxes, class_scores = model(images, training=True)

            dist_loss, class_loss = met(images, true_bboxes, true_labels, bboxes, class_scores, training=True)
            total_loss = dist_loss + class_loss

            if FLAGS.reg_loss_weight != 0:
                regex = r'.*(kernel|weight):0$'
                var_match = re.compile(regex)

                l2_loss = FLAGS.reg_loss_weight * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if var_match.match(v.name)])

                met.train_metric.reg_loss.update_state(l2_loss)

                total_loss += l2_loss

            met.train_metric.total_loss.update_state(total_loss)

        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(total_loss, model.trainable_variables)

        return total_loss, gradients

    def run_train_epoch(lr_sched, dataset, step_func, max_steps, broadcast_variables=False):
        step = 0
        acc_gradients = []
        for filenames, image_ids, images, true_bboxes, true_labels in dataset:
            new_lr = lr_sched(global_step.numpy())
            learning_rate.assign(new_lr)

            total_loss, gradients = train_step(filenames, images, true_bboxes, true_labels)

            if tf.math.is_nan(total_loss):
                logger.info('Loss is NaN, skipping training step')
            else:
                if len(acc_gradients) == 0:
                    acc_gradients = gradients
                else:
                    acc_gradients = [g1 + g2 for g1, g2 in zip(acc_gradients, gradients)]

                if FLAGS.grad_accumulate_steps <= 1 or (step + 1) % FLAGS.grad_accumulate_steps == 0:
                    mult = 1.
                    #if FLAGS.grad_accumulate_steps > 1:
                    #    mult = 1. / float(FLAGS.grad_accumulate_steps)

                    acc_gradients = [tf.clip_by_value(g * mult, -FLAGS.clip_grad, FLAGS.clip_grad) for g in acc_gradients]

                    opt.apply_gradients(zip(acc_gradients, model.trainable_variables))
                    acc_gradients = []

                if step == 0 and broadcast_variables:
                    logger.info('broadcasting initial variables')
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(opt.variables(), root_rank=0)
                    first_batch = False

            if (step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                logger.info('{}: step: {}/{} {}: total_loss: {:.3f}, {}'.format(
                    epoch_var.numpy(), step, max_steps, global_step.numpy(),
                    total_loss, met.str_result(training=True)
                    ))


            global_step.assign_add(1)
            step += 1
            if step >= max_steps:
                break

        logger.info('{}: step: {}/{} {}: total_loss: {:.3f}, {}'.format(
            epoch_var.numpy(), step, max_steps, global_step.numpy(),
            total_loss, met.str_result(training=True)
            ))

        if len(acc_gradients) > 0:
            opt.apply_gradients(zip(acc_gradients, model.trainable_variables))
            acc_gradients = []

        return step

    best_metric = 0
    best_saved_path = None

    if restore_path:
        met.reset_states()
        logger.info('there is a checkpoint {}, running initial validation'.format(restore_path))

        best_saved_path = restore_path

        best_metric = evaluate.evaluate(model, eval_dataset, class2idx, FLAGS.steps_per_eval_epoch)
        logger.info('initial validation metric: {:.3f}'.format(best_metric))

        if FLAGS.only_test:
            logger.info('Exiting...')
            exit(0)

    if best_metric < FLAGS.min_eval_metric:
        logger.info('setting minimal evaluation metric {:.4f} -> {} from command line arguments'.format(best_metric, FLAGS.min_eval_metric))
        best_metric = FLAGS.min_eval_metric

    num_vars = len(model.trainable_variables)
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

    logger.info('nodes: {}, checkpoint_dir: {}, model: D: {}, image_size: {}, model trainable variables/params: {}/{}'.format(
        num_replicas, checkpoint_dir, FLAGS.d, image_size,
        num_vars, int(num_params)))

    if FLAGS.lr_scheduler == 'cosine':
        decay_steps = FLAGS.num_epochs * steps_per_train_epoch
        if FLAGS.lr_decay_epochs > 0:
            decay_steps = FLAGS.lr_decay_epochs * steps_per_train_epoch

        lr_sched = callbacks.WarmupCosineDecayLRScheduler(max_lr=FLAGS.initial_learning_rate, min_lr=FLAGS.min_learning_rate,
                warmup_steps=FLAGS.lr_warmup_steps, decay_steps=decay_steps)
    elif FLAGS.lr_scheduler == 'reset':
        lr_sched = callbacks.ResetLRScheduler(learning_rate=FLAGS.initial_learning_rate, min_learning_rate=FLAGS.min_learning_rate,
                epochs_lr_update=FLAGS.epochs_lr_update, warmup_steps=FLAGS.lr_warmup_steps, reset_on_lr_update=FLAGS.reset_on_lr_update)

    for epoch in range(FLAGS.num_epochs):
        met.reset_states()

        if FLAGS.run_evaluation_first:
            new_metric = evaluate.evaluate(model, eval_dataset, class2idx, FLAGS.steps_per_eval_epoch)
            FLAGS.run_evaluation_first = False

        train_steps = run_train_epoch(lr_sched, train_dataset, train_step, FLAGS.steps_per_train_epoch, (epoch == 0))

        if hvd.rank() == 0:
            saved_path = manager.save()

        new_metric = evaluate.evaluate(model, eval_dataset, class2idx, FLAGS.steps_per_eval_epoch)

        logger.info('epoch: {}, train: steps: {}, lr: {:.2e}, train: {}, val_metric: {:.4f}/{:.4f}'.format(
            epoch_var.numpy(), global_step.numpy(),
            learning_rate.numpy(),
            met.str_result(True),
            new_metric, best_metric))

        if new_metric > best_metric:
            if hvd.rank() == 0:
                best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, new_metric))

            logger.info("epoch: {}, global_step: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}".format(
                epoch_var.numpy(), global_step.numpy(), best_saved_path, best_metric, new_metric))

            best_metric = new_metric

        want_reset = lr_sched.update(epoch_var.numpy(), global_step.numpy(), new_metric)

        if want_reset:
            restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
            if restore_path:
                epoch_num = epoch_var.numpy()
                step_num = global_step.numpy()
                logger.info('epoch: {}, global_step: {}, best metric: {:.5f}, learning rate: {:.2e} -> {:.2e}, restoring best checkpoint: {}'.format(
                    epoch_var.numpy(), global_step.numpy(), best_metric, learning_rate.numpy(), new_lr, best_saved_path))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

                checkpoint.restore(best_saved_path)

                epoch_var.assign(epoch_num)
                global_step.assign(step_num)

        # update learning rate even without resetting model
        learning_rate.assign(new_lr)
        epoch_var.assign_add(1)


if __name__ == '__main__':
    hvd.init()

    # always the same random seed
    random_seed = 913

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
