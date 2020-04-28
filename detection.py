import argparse
import cv2
import json
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa

import horovod.tensorflow as hvd

logger = logging.getLogger('detection')

import anchors
import autoaugment
import config
import encoder
import evaluate
import image as image_draw
import metric
import preprocess_ssd

parser = argparse.ArgumentParser()
parser.add_argument('--category_json', type=str, required=True, help='Category to ID mapping json file.')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
parser.add_argument('--eval_batch_size', type=int, default=128, help='Number of images to process in a batch.')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run.')
parser.add_argument('--epoch', type=int, default=0, help='Initial epoch\'s number')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs.')
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
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--dataset_type', type=str, choices=['tfrecords'], default='tfrecords', help='Dataset type')
parser.add_argument('--train_tfrecord_dir', type=str, help='Directory containing training TFRecords')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset')
parser.add_argument('--train_num_images', type=int, default=-1, help='Number of images in train epoch')
parser.add_argument('--eval_num_images', type=int, default=-1, help='Number of images in eval epoch')
parser.add_argument('--grad_accumulate_steps', type=int, default=1, help='Number of batches to accumulate before gradient update')
parser.add_argument('--use_random_augmentation', action='store_true', help='Use efficientnet random augmentation')
parser.add_argument('--is_mscoco_labels', action='store_true', help='Label ids start from 1, decrease it since ID must start from zero, this is not the case for Badoo dataset created by scan_tags.py script')

def unpack_tfrecord(serialized_example, image_size, num_classes, is_training, data_format):
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
    # labels should start from zero
    if FLAGS.is_mscoco_labels:
        orig_labels -= 1

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
        if FLAGS.use_random_augmentation:
            randaug_num_layers = 2
            randaug_magnitude = 28

            image = autoaugment.distort_image_with_randaugment(image, randaug_num_layers, randaug_magnitude)

        image, new_labels, new_bboxes = preprocess_ssd.preprocess_for_train(image, orig_labels, coords_yx,
                [image_size, image_size], data_format=data_format)

        yminf, xminf, ymaxf, xmaxf = tf.split(new_bboxes, num_or_size_splits=4, axis=1)
    else:
        image = preprocess_ssd.preprocess_for_eval(image, [image_size, image_size], data_format=data_format)
        new_labels = orig_labels

    xmin = xminf * image_size
    ymin = yminf * image_size
    xmax = xmaxf * image_size
    ymax = ymaxf * image_size
    new_bboxes = tf.concat([xmin, ymin, xmax, ymax], 1)

    return filename, image_id, image, new_bboxes, new_labels

def calc_epoch_steps(num_files, batch_size):
    return (num_files + batch_size - 1) // batch_size

def draw_bboxes(image_size, train_dataset, train_cat_names, all_anchors):
    data_dir = os.path.join(FLAGS.train_dir, 'tmp')
    os.makedirs(data_dir, exist_ok=True)

    all_anchors = all_anchors.numpy()

    for filename, image_id, image, true_values in train_dataset.unbatch().take(20):
        filename = str(filename.numpy(), 'utf8')

        dst = '{}/{}.png'.format(data_dir, image_id.numpy())

        true_values = true_values.numpy()

        non_background_index = np.where(true_values[..., 4] != 0)[0]
        #logger.info('{}: true_values: {}, non_background_index: {}'.format(filename, true_values.shape, non_background_index.shape))

        bboxes = true_values[non_background_index, 0:4]
        labels = true_values[non_background_index, 5:]
        labels = np.argmax(labels, axis=1)

        anchors = all_anchors[non_background_index, :]

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
            new_anns.append((_bb, l))

        logger.info('{}: true anchors: {}'.format(dst, len(new_anns)))

        image = image.numpy() * 128. + 128
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_draw.draw_im(image, new_anns, dst, train_cat_names)

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
    logger.info('FLAGS: {}'.format(FLAGS))

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
    learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

    model = encoder.create_model(FLAGS.d, FLAGS.num_classes)
    image_size = model.config.input_size

    anchors_all = generate_anchors(model.anchors_config, model.config.input_size)

    def create_dataset_from_tfrecord(name, dataset_dir, image_size, num_classes, is_training):
        filenames = []
        for fn in os.listdir(dataset_dir):
            fn = os.path.join(dataset_dir, fn)
            if os.path.isfile(fn):
                filenames.append(fn)

        if is_training and hvd.size() > 1 and len(filenames) > hvd.size():
            filenames = np.array_split(filenames, hvd.size())[hvd.rank()]

        ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
        ds = ds.map(lambda record: unpack_tfrecord(record,
                        image_size, num_classes, is_training,
                        FLAGS.data_format),
                num_parallel_calls=FLAGS.num_cpus)
        if is_training:
            ds = ds.shuffle(1024)
            batch_size = FLAGS.batch_size
        else:
            batch_size = FLAGS.eval_batch_size

        ds = ds.padded_batch(batch_size=batch_size,
                padded_shapes=((), (), (image_size, image_size, 3), (None, 4), (None,)),
                padding_values=('', tf.constant(0, dtype=tf.int64), 0., -1., -1))

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
        if not is_training:
            ds = ds.cache()

        logger.info('{} dataset has been created, tfrecords: {}'.format(name, len(filenames)))

        return ds

    if FLAGS.dataset_type == 'tfrecords':
        train_num_images = FLAGS.train_num_images
        eval_num_images = FLAGS.eval_num_images
        train_num_classes = FLAGS.num_classes
        train_cat_names = {}

        train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, image_size, train_num_classes, is_training=True)
        eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, image_size, train_num_classes, is_training=False)


    if False:
        draw_bboxes(image_size, train_dataset, train_cat_names, anchors_all)
        exit(0)

    if train_num_classes is None:
        logger.error('If there is no train_num_classes (tfrecord dataset), you must provide --num_classes')
        exit(-1)

    steps_per_train_epoch = FLAGS.steps_per_train_epoch
    if FLAGS.train_num_images > 0:
        steps_per_train_epoch = calc_epoch_steps(FLAGS.train_num_images, FLAGS.batch_size)

    steps_per_eval_epoch = FLAGS.steps_per_eval_epoch
    if FLAGS.eval_num_images > 0:
        steps_per_eval_epoch = calc_epoch_steps(FLAGS.eval_num_images, FLAGS.eval_batch_size)

    logger.info('steps_per_train_epoch: {}, train images: {}, steps_per_eval_epoch: {}, eval images: {}'.format(
        steps_per_train_epoch, train_num_images,
        steps_per_eval_epoch, eval_num_images))

    opt = tfa.optimizers.RectifiedAdam(lr=learning_rate, min_lr=FLAGS.min_learning_rate)
    opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
    if FLAGS.use_fp16:
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

    checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    status = checkpoint.restore(manager.latest_checkpoint)

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

    met = metric.ModelMetric(anchors_all, train_num_classes)
    epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf.function(experimental_relax_shapes=True)
    def train_step(filenames, images, true_bboxes, true_labels):
        with tf.GradientTape() as tape:
            bboxes, class_scores = model(images, training=True)

            dist_loss, class_loss = met(images, true_bboxes, true_labels, bboxes, class_scores, training=True)
            l2_loss = 1e-5 * sum([tf.reduce_sum(tf.pow(w, 2)) for w in model.trainable_variables])

            total_loss = dist_loss + class_loss + l2_loss

            met.train_metric.reg_loss.update_state(total_loss)
            met.train_metric.total_loss.update_state(total_loss)

        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(total_loss, model.trainable_variables)

        return total_loss, gradients

    def run_train_epoch(dataset, step_func, max_steps, broadcast_variables=False):
        step = 0
        acc_gradients = []
        for filenames, image_ids, images, true_bboxes, true_labels in dataset:
            total_loss, gradients = train_step(filenames, images, true_bboxes, true_labels)

            if tf.math.is_nan(total_loss):
                logger.info('Loss is NaN, skipping training step')
            else:
                if len(acc_gradients) == 0:
                    acc_gradients = gradients
                else:
                    acc_gradients = [g1+g2 for g1, g2 in zip(acc_gradients, gradients)]

                if FLAGS.grad_accumulate_steps <= 1 or (step + 1) % FLAGS.grad_accumulate_steps == 0:
                    opt.apply_gradients(zip(acc_gradients, model.trainable_variables))
                    acc_gradients = []

                if step == 0 and broadcast_variables:
                    logger.info('broadcasting initial variables')
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(opt.variables(), root_rank=0)
                    first_batch = False

            if (step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                logger.info('{}: step: {}/{} {}: total_loss: {:.3f}, {}'.format(
                    int(epoch_var.numpy()), step, max_steps, global_step.numpy(),
                    total_loss, met.str_result(training=True)
                    ))


            global_step.assign_add(1)
            step += 1
            if step >= max_steps:
                break

        if len(acc_gradients) > 0:
            opt.apply_gradients(zip(acc_gradients, model.trainable_variables))
            acc_gradients = []

        return step

    best_metric = 0
    best_saved_path = None
    num_epochs_without_improvement = 0
    initial_learning_rate_multiplier = 0.2
    learning_rate_multiplier = initial_learning_rate_multiplier

    if hvd.rank() == 0:
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

    learning_rate.assign(FLAGS.initial_learning_rate)
    for epoch in range(FLAGS.num_epochs):
        met.reset_states()
        want_reset = False

        train_steps = run_train_epoch(train_dataset, train_step, FLAGS.steps_per_train_epoch, (epoch == 0))

        if hvd.rank() == 0:
            saved_path = manager.save()

        new_metric = evaluate.evaluate(model, eval_dataset, class2idx, FLAGS.steps_per_eval_epoch)

        new_lr = learning_rate.numpy()

        logger.info('epoch: {}/{}, train: steps: {}, lr: {:.2e}, train: {}, val_metric: {:.4f}/{:.4f}'.format(
            int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(),
            learning_rate.numpy(),
            met.str_result(True),
            new_metric, best_metric))

        if new_metric > best_metric:
            if epoch_var.numpy() > FLAGS.skip_saving_epochs:
                if hvd.rank() == 0:
                    best_saved_path = checkpoint.save(file_prefix='{}/ckpt-{:.4f}'.format(good_checkpoint_dir, new_metric))

                logger.info("epoch: {}, global_step: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}".format(
                    int(epoch_var.numpy()), global_step.numpy(), best_saved_path, best_metric, new_metric))

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

                logger.info('epoch: {}/{}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))
                num_epochs_without_improvement = 0
                if learning_rate_multiplier > 0.1:
                    learning_rate_multiplier /= 2


            elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                new_lr = FLAGS.initial_learning_rate
                want_reset = True

                logger.info('epoch: {}/{}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier

        if want_reset:
            restore_path = tf.train.latest_checkpoint(good_checkpoint_dir)
            if restore_path:
                epoch_num = epoch_var.numpy()
                step_num = global_step.numpy()
                logger.info('epoch: {}/{}, global_step: {}, best metric: {:.5f}, learning rate: {:.2e} -> {:.2e}, restoring best checkpoint: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), best_metric, learning_rate.numpy(), new_lr, best_saved_path))

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
