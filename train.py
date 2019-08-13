import argparse
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf

import coco
import efficientnet
import image as image_draw
import preprocess
import ssd

from tensorflow.keras import Model
import tensorflow.keras.layers as layers

from PIL import Image


logger = logging.getLogger('objdet')
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
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to run.')
parser.add_argument('--epoch', type=int, default=0, help='Initial epoch\'s number')
parser.add_argument('--num_cpus', type=int, default=4, help='Number of parallel preprocessing jobs.')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored.')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--moving_average_decay', default=0, type=float, help='Moving average decay rate')
parser.add_argument('--batch_norm_momentum', default=None, type=float, help='Override batch normalization layer\'s momentum')
parser.add_argument('--batch_norm_epsilon', default=None, type=float, help='Override batch normalization layer\'s epsilon')
parser.add_argument('--dropout_rate', default=None, type=float, help='Dropout rate for the final output layer')
parser.add_argument('--drop_connect_rate', default=None, type=float, help='Drop connect rate for the network')
parser.add_argument('--depth_coefficient', default=None, type=float, help='Depth coefficient for scaling number of layers')
parser.add_argument('--width_coefficient', default=None, type=float, help='Width coefficient for scaling number of layers')
parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing parameter used in the softmax_cross_entropy')
parser.add_argument('--initial_learning_rate', default=1.6e-2, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--steps_per_eval', default=-1, type=int, help='Number of steps per evaluation run')
parser.add_argument('--steps_per_epoch', default=3000, type=int, help='Number of steps per training run')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
autoaugment_name_choice = ['v0']
parser.add_argument('--autoaugment_name', type=str, choices=autoaugment_name_choice, help='Autoaugment name, choices: {}'.format(autoaugment_name_choice))
FLAGS = parser.parse_args()

def tf_read_image(filename, anns, image_size, is_training, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_height = tf.shape(image)[0]
    orig_width = tf.shape(image)[1]

    image = preprocess.simple_resize_image(image, [image_size, image_size])

    fanns = []
    for bb, c in anns:
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        fbb = [x1/orig_width, y1/orig_height, x2/orig_width, y2/orig_height, float(c)]
        fanns.append(fbb)


def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def local_swish(x):
    return x * tf.nn.sigmoid(x)

def intersect(box_a, box_b):
    max_xy = tf.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = tf.maximum(box_a[:, None, :2], box_b[None, :, :2])
    #inter = tf.clip_by_value((max_xy - min_xy), 0, 10000000)
    inter = tf.nn.relu(max_xy - min_xy)
    return inter[:, :, 0] * inter[:, :, 1]

def box_sz(b):
    x = b[:, 2] - b[:, 0]
    y = b[:, 3] - b[:, 1]
    return x * y

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = tf.expand_dims(box_sz(box_a), 1) + tf.expand_dims(box_b, 0) - inter
    return inter / union

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]

    num_replicas = 1
    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
    #if True:
        FLAGS.initial_learning_rate *= num_replicas
        FLAGS.batch_size *= num_replicas

        learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

        params = {
            'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
            'data_format': FLAGS.data_format,
            'relu_fn': local_swish
        }

        if FLAGS.batch_norm_momentum is not None:
            params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
        if FLAGS.batch_norm_epsilon is not None:
            params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
        if FLAGS.dropout_rate is not None:
            params['dropout_rate'] = FLAGS.dropout_rate
        if FLAGS.drop_connect_rate is not None:
            params['drop_connect_rate'] = FLAGS.drop_connect_rate
        if FLAGS.depth_coefficient:
            params['depth_coefficient'] = FLAGS.depth_coefficient
        if FLAGS.width_coefficient:
            params['width_coefficient'] = FLAGS.width_coefficient

        output_endpoints = []

        global_step = tf.Variable(1, dtype=tf.int64, name='global_step')

        base_model = efficientnet.build_model(model_name=FLAGS.model_name, override_params=params)
        base_output = base_model(tf.zeros((1, image_size, image_size, 3)), features_only=True, training=True)
        for name, endpoint in base_model.endpoints.items():
            m = re.match('reduction_[\d]+$', name)
            if not m:
                continue

            shape = endpoint.shape
            h = shape[1]
            w = shape[2]

            if w <= 7 or h <= 7:
                output_endpoints.append((name, endpoint))

        output_endpoints.append(('final_features', base_model.endpoints['features']))

        base_num_params = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, base model trainable variables: {}, trainable params: {}, dtype: {}, autoaugment_name: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size, len(base_model.trainable_variables), int(base_num_params), dtype, FLAGS.autoaugment_name))

        endpoint_shapes = [e.shape for name, e in output_endpoints]
        logger.info('output endpoints: {}'.format(endpoint_shapes))

        has_moving_average_decay = (FLAGS.moving_average_decay > 0)

        restore_vars_dict = None

        if has_moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(decay=FLAGS.moving_average_decay, num_updates=global_step)

            ema_vars = tf.trainable_variables() + tf.compat.v1.get_collection('moving_vars')
            for v in tf.global_variables():
                # We maintain mva for batch norm moving mean and variance as well.
                if 'moving_mean' in v.name or 'moving_variance' in v.name:
                    ema_vars.append(v)

            ema_vars = list(set(ema_vars))

        def create_dataset(name, ann_file, data_dir, is_training):
            ds = coco.COCO(ann_file, data_dir)
            def gen():
                for filename, image_id, anns in zip(*ds.get_images()):
                    orig_im = Image.open(filename)
                    orig_width = orig_im.width
                    orig_height = orig_im.height

                    if orig_im.mode != "RGB":
                        orig_im = orig_im.convert("RGB")

                    size = (image_size, image_size)
                    im = orig_im.resize(size, resample=Image.BILINEAR)

                    a = np.asarray(im)

                    fanns = []
                    for bb, c in anns:
                        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                        fbb = [x1/orig_width, y1/orig_height, x2/orig_width, y2/orig_height, float(c)]
                        fanns.append(fbb)

                    logger.info('{}: {} {} -> {}, anns: {}'.format(filename, a.shape, orig_im.size, im.size, len(fanns)))

                    yield filename, image_id, a, fanns[:1]


            dataset = tf.data.Dataset.from_generator(gen,
                                                     output_types=(tf.string, tf.int32, tf.uint8, dtype),
                                                     output_shapes=(tf.TensorShape([]),
                                                                    tf.TensorShape([]),
                                                                    tf.TensorShape([image_size, image_size, 3]),
                                                                    tf.TensorShape([None, 5])))

            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(FLAGS.batch_size * 2)
            dataset = dataset.batch(FLAGS.batch_size)
            dataset = dataset.repeat()

            #logger.info('{}: dataset has been created, filename: {}, images: {}, categories: {}'.format(name, ann_file, ds.num_images(), ds.num_classes()))

            return dataset, ds.num_images(), ds.num_classes(), ds.cat_names()

        train_dataset, train_num_images, train_num_classes, cat_names = create_dataset('train', FLAGS.train_coco_annotations, FLAGS.train_coco_data_dir, is_training=True)
        eval_dataset, eval_num_images, eval_num_classes, cat_names = create_dataset('eval', FLAGS.eval_coco_annotations, FLAGS.eval_coco_data_dir, is_training=False)

        if False:
            data_dir = os.path.join(FLAGS.train_dir, 'tmp')
            os.makedirs(data_dir, exist_ok=True)

            for filename, image_id, image, anns in train_dataset.take(1).unbatch():
                h = tf.cast(tf.shape(image)[0], tf.float32)
                w = tf.cast(tf.shape(image)[1], tf.float32)

                filename = str(filename)
                new_anns = []
                for ann in anns:

                    x1, y1, x2, y2, c = ann[0], ann[1], ann[2], ann[3], ann[4]
                    x1 *= w
                    x2 *= w
                    y1 *= h
                    y2 *= h

                    nbb = [x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy()]
                    new_anns.append((nbb, int(c)))

                filename = str(filename)

                dst = '{}/{}.jpg'.format(data_dir, image_id)
                image_draw.draw_im(image.numpy(), new_anns, dst, cat_names)


        num_anchors = 1
        model = ssd.SSD(base_model._global_params, output_endpoints, num_anchors, train_num_classes)

        _ = model(base_model, tf.zeros((1, image_size, image_size, 3)), training=True)


        steps_per_epoch = calc_epoch_steps(train_num_images)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_num_images)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        if FLAGS.checkpoint:
            status = checkpoint.restore(FLAGS.checkpoint)
            status.expect_partial()

            logger.info("Restored from external checkpoint {}".format(manager.latest_checkpoint))

        status = checkpoint.restore(manager.latest_checkpoint)
        status.expect_partial()

        if manager.latest_checkpoint:
            logger.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

        reg_loss_metric = tf.keras.metrics.Mean(name='reg_train_loss')
        ce_loss_metric = tf.keras.metrics.Mean(name='ce_train_loss')
        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        eval_ce_loss_metric = tf.keras.metrics.Mean(name='ce_eval_loss')
        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

        def reset_metrics():
            reg_loss_metric.reset_states()
            ce_loss_metric.reset_states()
            loss_metric.reset_states()
            accuracy_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_ce_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()


        cross_entropy_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=FLAGS.label_smoothing)

        def calculate_metrics(logits, labels):
            # Calculate loss, which includes softmax cross entropy and L2 regularization.
            one_hot_labels = tf.one_hot(labels, num_classes)

            ce_loss = cross_entropy_loss_object(y_pred=logits, y_true=one_hot_labels)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

            reg_loss = 0
            # Add weight decay to the loss for non-batch-normalization variables.
            reg_loss = FLAGS.reg_weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'batch_normalization' not in v.name])
            reg_loss = tf.nn.scale_regularization_loss(reg_loss)

            total_loss = ce_loss + reg_loss

            return reg_loss, ce_loss, total_loss

        def eval_step(images, labels):
            logits = model(images, training=False)
            reg_loss, ce_loss, total_loss = calculate_metrics(logits, labels)

            eval_loss_metric.update_state(total_loss)
            eval_ce_loss_metric.update_state(ce_loss)
            eval_accuracy_metric.update_state(labels, logits)

            return reg_loss, ce_loss, total_loss

        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                reg_loss, ce_loss, total_loss = calculate_metrics(logits, labels)

            variables = model.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            opt.apply_gradients(zip(gradients, variables))

            loss_metric.update_state(total_loss)
            reg_loss_metric.update_state(reg_loss)
            ce_loss_metric.update_state(ce_loss)
            accuracy_metric.update_state(labels, logits)

            global_step.assign_add(1)

            return reg_loss, ce_loss, total_loss

        if has_moving_average_decay:
            with tf.control_dependencies([train_op]):
                train_op = ema.apply(ema_vars)

            # Load moving average variables for eval.
            #restore_vars_dict = ema.variables_to_restore(ema_vars)

        steps_per_epoch = calc_epoch_steps(train_bg)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_bg)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch: {}/{}, steps_per_eval: {}/{}'.format(steps_per_epoch, calc_epoch_steps(train_bg), steps_per_eval, calc_epoch_steps(eval_bg)))

        @tf.function
        def distributed_step(step_func, images, labels):
            pr_reg_losses, pr_ce_losses, pr_total_losses = dstrategy.experimental_run_v2(step_func, args=(images, labels,))

            #reg_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_reg_losses, axis=None)
            #ce_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_ce_losses, axis=None)
            #total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)

        def run_epoch(bg, dataset, step_func, max_steps):
            losses = []
            accs = []

            step = 0
            for images, labels, filenames in dataset:
                # In most cases, the default data format NCHW instead of NHWC should be
                # used for a significant performance boost on GPU/TPU. NHWC should be used
                # only if the network needs to be run on CPU since the pooling operations
                # are only supported on NHWC.
                if FLAGS.data_format == 'channels_first':
                    images = tf.transpose(images, [0, 3, 1, 2])


                #step_func(images, labels)
                distributed_step(step_func, images, labels)

                step += 1
                if step >= max_steps:
                    break

            return step

        min_metric = 0
        num_epochs_without_improvement = 0

        if manager.latest_checkpoint:
            reset_metrics()
            logger.info('there is a checkpoint {}, running initial validation'.format(manager.latest_checkpoint))

            # we have to call training @tf.function distributed_step() first, since it is the only time when variables can be created,
            # if distributed_step() with eval function is called first, then no training model weights will be created and subsequent training will fail
            # maybe the right solution is to separate train functions (as steps already are) and call them with their own datasets
            _ = run_epoch(train_bg, train_dataset, train_step, 0)
            eval_steps = run_epoch(eval_bg, eval_dataset, eval_step, steps_per_eval)
            min_metric = eval_accuracy_metric.result()
            logger.info('initial validation metric: {}'.format(min_metric))

        if min_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {} -> {} from command line arguments'.format(min_metric, FLAGS.min_eval_metric))
            min_metric = FLAGS.min_eval_metric

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            reset_metrics()

            train_steps = run_epoch(train_bg, train_dataset, train_step, steps_per_epoch)
            eval_steps = run_epoch(eval_bg, eval_dataset, eval_step, steps_per_eval)

            logger.info('epoch: {}, train: steps: {}, reg_loss: {:.2e}, ce_loss: {:.2e}, total_loss: {:.2e}, accuracy: {:.5f}, eval: ce_loss: {:.2e}, total_loss: {:.2e}, accuracy: {:.5f}, lr: {:.2e}'.format(
                epoch, global_step.numpy(),
                reg_loss_metric.result(), ce_loss_metric.result(), loss_metric.result(), accuracy_metric.result(),
                eval_ce_loss_metric.result(), eval_loss_metric.result(), eval_accuracy_metric.result(),
                learning_rate.numpy()))

            eval_acc = eval_accuracy_metric.result()
            if eval_acc > min_metric:
                save_path = manager.save()
                logger.info("epoch: {}, saved checkpoint: {}, eval accuracy: {:.5f} -> {:.5f}".format(epoch, save_path, min_metric, eval_acc))
                min_metric = eval_acc
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement > 10:
                want_reset = False

                if learning_rate > FLAGS.min_learning_rate:
                    new_lr = learning_rate.numpy() / 5.
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, min_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    want_reset = True
                elif num_epochs_without_improvement > 20:
                    new_lr = FLAGS.initial_learning_rate
                    logger.info('epoch: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}'.format(
                        epoch, num_epochs_without_improvement, min_metric, learning_rate.numpy(), new_lr))
                    learning_rate.assign(new_lr)
                    num_epochs_without_improvement = 0
                    want_reset = True

                if want_reset:
                    logger.info('epoch: {}, best metric: {:.5f}, learning rate: {:.2e}, restoring best checkpoint: {}'.format(
                        epoch, min_metric, learning_rate.numpy(), manager.latest_checkpoint))

                    checkpoint.restore(manager.latest_checkpoint)

def main():
    try:
        train()
    except Exception as e: #pylint: disable=W0703
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)

if __name__ == '__main__':
    main()
