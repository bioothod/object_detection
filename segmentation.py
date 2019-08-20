import argparse
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


import efficientnet
import image as image_draw
import loss
import polygon_dataset
import preprocess
import validate
import unet

logger = logging.getLogger('segmentation')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, required=True, help='Path to train image+annotation directory')
parser.add_argument('--eval_data_dir', type=str, help='Path to eval image+annotation directory')
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run.')
parser.add_argument('--epoch', type=int, default=0, help='Initial epoch\'s number')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs.')
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
parser.add_argument('--steps_per_epoch', default=-1, type=int, help='Number of steps per training run')
parser.add_argument('--min_eval_metric', default=0.75, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
autoaugment_name_choice = ['v0']
parser.add_argument('--autoaugment_name', type=str, choices=autoaugment_name_choice, help='Autoaugment name, choices: {}'.format(autoaugment_name_choice))
parser.add_argument('--dataset', type=str, choices=['card_images', 'oxford_pets'], default='card_images', help='Dataset type')
FLAGS = parser.parse_args()

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def local_swish(x):
    return x * tf.nn.sigmoid(x)

@tf.function
def basic_preprocess(filename, mask, image_size, is_training, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.processing_function(image, image_size, image_size, is_training, dtype)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    return filename, image, mask


@tf.function
def call_base_model(model, inputs):
    #return model(inputs, training=True, features_only=True)
    return model(inputs, training=True)

@tf.function
def call_model(model, inputs):
    return model(inputs, training=True)

@tf.function
def load_image(datapoint, image_size, is_training, dtype):
    image = datapoint['image']
    mask = datapoint['segmentation_mask']
    filename = datapoint['file_name']

    image = preprocess.processing_function(image, image_size, image_size, is_training, dtype)
    mask = preprocess.try_resize(image, image_size, image_size) - 1

    if tf.random.uniform(()) > 0.5 and is_training:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    return filename, image, mask

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    #image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]
    image_size = 128

    num_replicas = 1
    #dstrategy = None
    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
    #if True:
        FLAGS.initial_learning_rate *= num_replicas
        FLAGS.batch_size *= num_replicas

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

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        global_step = tf.Variable(1, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')

        base_model = tf.keras.applications.MobileNetV2(input_shape=[image_size, image_size, 3], include_top=False)
        base_model.trainable = False

        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        layers = [base_model.get_layer(name).output for name in layer_names]
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        up_stack = [
            unet.upsample(512, 3),  # 4x4 -> 8x8
            unet.upsample(256, 3),  # 8x8 -> 16x16
            unet.upsample(128, 3),  # 16x16 -> 32x32
            unet.upsample(64, 3),   # 32x32 -> 64x64
        ]

        num_classes = 3
        last = tf.keras.layers.Conv2DTranspose(num_classes, 3, strides=2, padding='same', activation='softmax')

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            upsampled = up(x)
            logger.info('x: {}, up: {}, skip: {}'.format(x.shape, upsampled.shape, skip.shape))

            concat = tf.keras.layers.Concatenate()
            x = concat([upsampled, skip])

        ret = last(x)
        model = tf.keras.Model(inputs=inputs, outputs=ret)
        #base_model = efficientnet.build_model(FLAGS.model_name, params)
        #if dstrategy is not None:
        #    dstrategy.experimental_run_v2(call_base_model, args=(base_model, tf.ones((1, image_size, image_size, 3))))

        #model = unet.Unet(3, down_stack)
        #if dstrategy is not None:
        #    dstrategy.experimental_run_v2(call_model, args=(model, tf.ones((1, image_size, image_size, 3))))
        #else:
        #    dummy_init_unet(model, 1, image_size)

        num_params_base = 0
        num_params_unet = 0
        num_vars = 0
        #num_vars = len(model.trainable_variables)
        #num_params_base = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])
        #num_params_unet = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, base model trainable variables/params: {}/{}, unet model trainable variables/params: {}/{}, dtype: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
            len(base_model.trainable_variables), int(num_params_base),
            num_vars, int(num_params_unet),
            dtype))

        def create_dataset(ann_dir, is_training):
            cards = polygon_dataset.Polygons(ann_dir, logger, image_size, image_size)

            dataset = tf.data.Dataset.from_tensor_slices((cards.get_filenames(), cards.get_masks()))
            dataset = dataset.map(lambda filename, mask: basic_preprocess(filename, mask, image_size, is_training, dtype), num_parallel_calls=FLAGS.num_cpus)

            logger.info('dataset has been created, data dir: {}, images: {}'.format(ann_dir, cards.num_images()))

            return dataset, cards.num_images()

        if FLAGS.dataset == 'card_images':
            dataset, num_images = create_dataset(FLAGS.train_data_dir, is_training=True)

            rate = 0.9
            eval_num_images = int(num_images * (1. - rate))
            train_num_images = num_images - eval_num_images

            train_dataset = dataset.skip(eval_num_images).cache().shuffle(FLAGS.batch_size * 2).batch(FLAGS.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            train_dataset = dstrategy.experimental_distribute_dataset(train_dataset)

            eval_dataset = dataset.take(eval_num_images).cache().batch(FLAGS.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            eval_dataset = dstrategy.experimental_distribute_dataset(eval_dataset)
        elif FLAGS.dataset == 'oxford_pets':
            dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True, data_dir=FLAGS.train_data_dir)

            train_dataset = dataset['train'].map(lambda datapoint: load_image(datapoint, image_size, True, dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_dataset = train_dataset.shuffle(FLAGS.batch_size * 2).batch(FLAGS.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #train_dataset = dstrategy.experimental_distribute_dataset(train_dataset)
            train_num_images = info.splits['train'].num_examples

            eval_dataset = dataset['test'].map(lambda datapoint: load_image(datapoint, image_size, False, dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            eval_dataset = eval_dataset.batch(FLAGS.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #eval_dataset = dstrategy.experimental_distribute_dataset(eval_dataset)
            eval_num_images = info.splits['test'].num_examples

        steps_per_epoch = calc_epoch_steps(train_num_images)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_num_images)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch: {}/{}, train images: {}, steps_per_eval: {}/{}, eval images: {}'.format(
            steps_per_epoch, calc_epoch_steps(train_num_images), train_num_images,
            steps_per_eval, calc_epoch_steps(eval_num_images), eval_num_images))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        want_keras = False
        if want_keras:
            model.compile(optimizer=opt, loss=[tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)], metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

            @tf.function
            def keras_dataset(filename, image, mask):
                return image, mask
            train_dataset = train_dataset.unbatch().map(keras_dataset).batch(FLAGS.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            eval_dataset = eval_dataset.unbatch().map(keras_dataset).batch(FLAGS.batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            model_history = model.fit(train_dataset, epochs=FLAGS.num_epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=steps_per_eval,
                          validation_data=eval_dataset)
            exit(0)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        if FLAGS.checkpoint:
            status = checkpoint.restore(FLAGS.checkpoint)
            status.expect_partial()

            logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))

        status = checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            logger.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        eval_loss_metric = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

        def reset_metrics():
            loss_metric.reset_states()
            accuracy_metric.reset_states()

            eval_loss_metric.reset_states()
            eval_accuracy_metric.reset_states()

        num_classes = 3
        #cross_entropy_loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, label_smoothing=FLAGS.label_smoothing)
        cross_entropy_loss_object = loss.FocalLoss(reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(logits, labels):
            one_hot_labels = tf.one_hot(labels, num_classes, axis=3)
            one_hot_labels = tf.squeeze(one_hot_labels, axis=4)

            ce_loss = cross_entropy_loss_object(y_pred=logits, y_true=one_hot_labels)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

            total_loss = ce_loss
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
            gradients = [tf.clip_by_value(g, -1, 1) for g in gradients]
            opt.apply_gradients(zip(gradients, variables))

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

                if False:
                    @tf.function
                    def eval_step_mask(images):
                        logits = model(images, training=False)
                        masks = tf.argmax(logits, axis=-1)
                        return masks

                    dst_dir = os.path.join(FLAGS.train_dir, 'eval', str(epoch))
                    os.makedirs(dst_dir, exist_ok=True)

                    num_files = 0
                    for filenames, images, true_masks in eval_dataset:
                        masks = eval_step_mask(images)
                        num_files += len(filenames)

                        generate_images(filenames, images, masks, dst_dir)
                        logger.info('saved {}/{} images'.format(len(filenames), total_images))

                        if num_files >= eval_num_images:
                            break
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
