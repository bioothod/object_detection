import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

import anchors_gen
import coco
import image as image_draw
import loss
import map_iter
import preprocess_ssd
import ssd

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
FLAGS = parser.parse_args()


def unpack_tfrecord(serialized_example, np_anchor_boxes, np_anchor_areas, image_size, is_training):
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

    if FLAGS.orig_images:
        cx, cy, h, w = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)
        cx = tf.squeeze(cx, 1)
        cy = tf.squeeze(cy, 1)
        h = tf.squeeze(h, 1)
        w = tf.squeeze(w, 1)

        image_height = tf.cast(tf.shape(image)[0], tf.float32)
        image_width = tf.cast(tf.shape(image)[1], tf.float32)

        xmin = (cx - w/2) / image_width
        xmax = (cx + w/2) / image_width
        ymin = (cy - h/2) / image_height
        ymax = (cy + h/2) / image_height

        coords_yx = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        if is_training:
            image, orig_labels, orig_bboxes = preprocess_ssd.preprocess_for_train(image, orig_labels, coords_yx, [image_size, image_size], data_format=FLAGS.data_format)

            xminf, yminf, xmaxf, ymaxf = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)
            #xminf = tf.squeeze(xminf, 1)
            #yminf = tf.squeeze(yminf, 1)
            #xmaxf = tf.squeeze(xaxnf, 1)
            #ymaxf = tf.squeeze(ymaxf, 1)
            cx = (xminf + xmaxf) * image_size / 2
            cy = (yminf + ymaxf) * image_size / 2
            h = (ymaxf - yminf) * image_size / 2
            w = (xmaxf - xminf) * image_size / 2

            orig_bboxes = tf.concat([cx, cy, h, w], axis=1)
        else:
            image = preprocess_ssd.preprocess_for_eval(image, [image_size, image_size], data_format=FLAGS.data_format)
    else:
        image -= 128.

    image /= 128.

    true_bboxes, true_labels = anchors_gen.generate_true_labels_for_anchors(orig_bboxes, orig_labels, np_anchor_boxes, np_anchor_areas)

    return filename, image_id, image, true_bboxes, true_labels
    #return filename, image_id, image, orig_bboxes, orig_labels

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1), square_loss, absolute_loss-0.5)

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

        model, np_anchor_boxes, np_anchor_areas = ssd.create_model(dtype, FLAGS.model_name, train_base.num_classes())
        image_size = model.image_size
        num_anchors = np_anchor_boxes.shape[0]
        logger.info('base_model: {}, num_anchors: {}, image_size: {}'.format(FLAGS.model_name, num_anchors, image_size))

        if False:
            import pickle
            d = {
                'np_anchor_boxes': np_anchor_boxes,
                'np_anchor_areas': np_anchor_areas,
            }

            with open(os.path.join(logdir, 'dump.pickle'), 'wb') as f:
                pickle.dump(d, f, protocol=4)
            exit(0)

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

        def create_dataset_from_tfrecord(name, dataset_dir, is_training):
            filenames = []
            for fn in os.listdir(dataset_dir):
                fn = os.path.join(dataset_dir, fn)
                if os.path.isfile(fn):
                    filenames.append(fn)

            ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=2)
            ds = ds.map(lambda record: unpack_tfrecord(record, np_anchor_boxes, np_anchor_areas, image_size, is_training), num_parallel_calls=FLAGS.num_cpus)
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
            train_dataset = create_dataset_from_tfrecord('train', FLAGS.train_tfrecord_dir, is_training=True)
            eval_dataset = create_dataset_from_tfrecord('eval', FLAGS.eval_tfrecord_dir, is_training=False)

            train_num_images = FLAGS.train_num_images
            eval_num_images = FLAGS.eval_num_images
            train_num_classes = FLAGS.num_classes
            train_cat_names = {}

        if False:
            data_dir = os.path.join(FLAGS.train_dir, 'tmp')
            os.makedirs(data_dir, exist_ok=True)

            for filename, image_id, image, true_bboxes, true_labels in train_dataset.unbatch().take(20):
            #for filename, image_id, image, true_bboxes, true_labels in train_dataset.take(10):
                filename = str(filename.numpy(), 'utf8')
                true_labels = true_labels.numpy()
                true_bboxes = true_bboxes.numpy()

                dst = '{}/{}.png'.format(data_dir, image_id.numpy())
                new_anns = []
                # category ID in true_labels does not match train_cat_names here, this is converted category_id into range [0, num_categories], where 0 is background class
                # true_orig_labels contain original category ids
                non_background_index = np.where(true_labels != 0)

                #logger.info('{}: true_labels: {}, true_bboxes: {}, non_background_index: {}'.format(filename, true_labels, true_bboxes, non_background_index))

                for bb, cat_id in zip(true_bboxes[non_background_index], true_labels[non_background_index]):
                    cx, cy, h, w = bb
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
                image_draw.draw_im(image, new_anns, dst, train_cat_names)

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

        num_vars_ssd = len(model.trainable_variables)
        num_params_ssd = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, ssd model trainable variables/params: {}/{}, dtype: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
            num_vars_ssd, int(num_params_ssd),
            dtype))

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
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        full_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_full_accuracy')
        iou_metric = tf.keras.metrics.Mean(name='train_iou')
        num_good_ious_metric = tf.keras.metrics.Mean(name='eval_num_good_ious')
        num_positive_labels_metric = tf.keras.metrics.Mean(name='eval_num_positive_labels_ious')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_full_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_full_accuracy')
        eval_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
        eval_distance_center_metric = tf.keras.metrics.MeanAbsoluteError(name='eval_dist_center')
        eval_distance_size_metric = tf.keras.metrics.MeanAbsoluteError(name='eval_dist_size')
        eval_iou_metric = tf.keras.metrics.Mean(name='eval_iou')
        eval_num_good_ious_metric = tf.keras.metrics.Mean(name='eval_num_good_ious')
        eval_num_positive_labels_metric = tf.keras.metrics.Mean(name='eval_num_positive_labels_ious')

        def reset_metrics():
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            full_accuracy_metric.reset_states()
            iou_metric.reset_states()
            num_good_ious_metric.reset_states()
            num_positive_labels_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()
            eval_full_accuracy_metric.reset_states()
            eval_iou_metric.reset_states()
            eval_num_good_ious_metric.reset_states()
            eval_num_positive_labels_metric.reset_states()

        ce_loss_object = loss.CategoricalFocalLoss(reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(logits, bboxes, true_labels, loss_metric, accuracy_metric, full_accuracy_metric, iou_metric, num_good_ious_metric, num_positive_labels_metric):
            coords, classes = logits
            classes = tf.nn.softmax(classes, -1)

            positive_indexes = tf.where(true_labels > 0)
            num_positives = tf.shape(positive_indexes)[0]
            num_positive_labels_metric.update_state(num_positives)

            ce_loss, dist_loss, total_loss = 0., 0., 0.

            if num_positives > 0:
                num_negatives = tf.cast(FLAGS.negative_positive_rate * tf.cast(num_positives, tf.float32), tf.int32)

                # because 'x == 0' condition yields (none, 0) tensor, and ' < 1' yields (none, 2) shape, the same as aboe positive index selection
                negative_indexes = tf.where(true_labels < 1)
                # selecting scores for background class among true backgrounds (true_label == 0)
                #negative_pred_background = tf.gather_nd(classes, negative_indexes)[:, 0]

                negative_true = tf.gather_nd(true_labels, negative_indexes)
                negative_pred = tf.gather_nd(classes, negative_indexes)
                negative_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=negative_true, logits=negative_pred)

                num_negatives_to_sample = tf.minimum(num_negatives, tf.shape(negative_indexes)[0])
                #sampled_negative_pred_background, sampled_negative_indexes = tf.math.top_k(negative_pred_background, num_negatives_to_sample)
                sorted_negative_indexes = tf.argsort(negative_ce, direction='ASCENDING')
                sampled_negative_indexes = sorted_negative_indexes[:num_negatives_to_sample]

                negative_indexes = tf.gather(negative_indexes, sampled_negative_indexes)

                new_indexes = tf.concat([positive_indexes, negative_indexes], axis=0)

                sampled_true_labels_full = tf.gather_nd(true_labels, new_indexes)
                sampled_true_labels_one_hot = tf.one_hot(sampled_true_labels_full, train_num_classes, axis=-1)
                sampled_pred_classes_full = tf.gather_nd(classes, new_indexes)

                ce_loss = ce_loss_object(y_true=sampled_true_labels_one_hot, y_pred=sampled_pred_classes_full)
                ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

                sampled_true_labels_pos = tf.gather_nd(true_labels, positive_indexes)
                sampled_pred_classes_pos = tf.gather_nd(classes, positive_indexes)

                sampled_true_bboxes_pos = tf.gather_nd(bboxes, positive_indexes)
                sampled_pred_coords_pos = tf.gather_nd(coords, positive_indexes)

                tcx, tcy, th, tw = tf.split(sampled_true_bboxes_pos, num_or_size_splits=4, axis=1)
                pcx, pcy, ph, pw = tf.split(sampled_pred_coords_pos, num_or_size_splits=4, axis=1)

                pw += 1e-10
                ph += 1e-10

                dc = tf.math.abs(tcx - pcx) / pw + tf.math.abs(tcy - pcy) / ph
                dc *= 5

                ds = tf.math.log(tw / pw) + tf.math.log(th / ph)
                ds *= 10

                dist_loss = smooth_l1_loss(dc + ds)
                dist_loss = tf.nn.compute_average_loss(dist_loss, global_batch_size=FLAGS.batch_size)

                total_loss = ce_loss + dist_loss
                loss_metric.update_state(total_loss)

                # metrics update

                full_accuracy_metric.update_state(y_true=sampled_true_labels_full, y_pred=sampled_pred_classes_full)
                accuracy_metric.update_state(y_true=sampled_true_labels_pos, y_pred=sampled_pred_classes_pos)

                ious = anchors_gen.calc_ious_one_to_one(sampled_pred_coords_pos, sampled_true_bboxes_pos)
                good_iou_index = tf.where(ious > 0.5)
                num_good_ious = tf.shape(good_iou_index)[0]

                iou_metric.update_state(ious)
                num_good_ious_metric.update_state(num_good_ious)

            return ce_loss, dist_loss, total_loss

        epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        def eval_step(filenames, images, bboxes, true_labels):
            logits = model(images, training=False)
            ce_loss, dist_loss, total_loss = calculate_metrics(logits, bboxes, true_labels,
                    eval_loss_metric, eval_accuracy_metric, eval_full_accuracy_metric, eval_iou_metric,
                    eval_num_good_ious_metric, eval_num_positive_labels_metric)
            return ce_loss, dist_loss, total_loss

        def train_step(filenames, images, bboxes, true_labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                ce_loss, dist_loss, total_loss = calculate_metrics(logits, bboxes, true_labels,
                        loss_metric, accuracy_metric, full_accuracy_metric, iou_metric,
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

            return ce_loss, dist_loss, total_loss

        @tf.function
        def distributed_train_step(args):
            pr_ce_losses, pr_dist_losses, pr_total_losses = dstrategy.experimental_run_v2(train_step, args=args)
            ce_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_ce_losses, axis=None)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_dist_losses, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)
            return ce_loss, dist_loss, total_loss

        @tf.function
        def distributed_eval_step(args):
            pr_ce_losses, pr_dist_losses, pr_total_losses = dstrategy.experimental_run_v2(eval_step, args=args)
            ce_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_ce_losses, axis=None)
            dist_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_dist_losses, axis=None)
            total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)
            return ce_loss, dist_loss, total_loss

        def run_epoch(name, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for filenames, image_ids, images, bboxes, true_labels in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                ce_loss, dist_loss, total_loss = step_func(args=(filenames, images, bboxes, true_labels))
                if name == 'train' and step % FLAGS.print_per_train_steps == 0:
                    logger.info('{}: {}: step: {}/{}, ce_loss: {:.2e}, dist_loss: {:.2e}, total_loss: {:.2e}, accuracy: {:.3f}/{:.3f}, iou: {:.3f}, good_ios/pos: {}/{}'.format(
                        name, epoch_var.numpy(), step, max_steps, ce_loss, dist_loss, total_loss,
                        accuracy_metric.result(), full_accuracy_metric.result(),
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

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            epoch_var.assign(epoch)

            reset_metrics()

            train_steps = run_epoch('train', train_dataset, distributed_train_step, steps_per_epoch)
            eval_steps = run_epoch('eval', eval_dataset, distributed_eval_step, steps_per_eval)

            metric = validation_metric()

            logger.info('epoch: {}, train: steps: {}, accuracy: {:.3f}/{:.3f}, iou: {:.3f}, good_ios/pos: {}/{}, loss: {:.2e}, eval: accuracy: {:.3f}/{:.3f}, iou: {:.3f}, good_ios/pos: {}/{}, loss: {:.2e}, lr: {:.2e}, val_metric: {:.3f}'.format(
                epoch, global_step.numpy(),
                accuracy_metric.result(), full_accuracy_metric.result(), iou_metric.result(),
                int(num_good_ious_metric.result()), int(num_positive_labels_metric.result()),
                loss_metric.result(),
                eval_accuracy_metric.result(), eval_full_accuracy_metric.result(), eval_iou_metric.result(),
                int(eval_num_good_ious_metric.result()), int(eval_num_positive_labels_metric.result()),
                eval_loss_metric.result(),
                learning_rate.numpy(),
                metric))

            if metric > min_metric:
                save_path = manager.save()
                logger.info("epoch: {}, saved checkpoint: {}, eval metric: {:.4f} -> {:.4f}, accuracy: {:.3f}/{:.3f}, iou: {:.3f}, good_ios/positive: {}/{}".format(
                    epoch, save_path, min_metric, metric, 
                    eval_accuracy_metric.result(), eval_full_accuracy_metric.result(), eval_iou_metric.result(),
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
