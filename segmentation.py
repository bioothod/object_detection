import argparse
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf

import efficientnet
import image as image_draw
import polygon_dataset

from tensorflow.keras import Model
from tensorflow_examples.models.pix2pix import pix2pix

from PIL import Image


logger = logging.getLogger('segmentation')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, required=True, help='Path to train image+annotation directory')
parser.add_argument('--eval_data_dir', type=str, required=True, help='Path to eval image+annotation directory')
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
parser.add_argument('--steps_per_epoch', default=-1, type=int, help='Number of steps per training run')
parser.add_argument('--min_eval_metric', default=0.75, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
autoaugment_name_choice = ['v0']
parser.add_argument('--autoaugment_name', type=str, choices=autoaugment_name_choice, help='Autoaugment name, choices: {}'.format(autoaugment_name_choice))
FLAGS = parser.parse_args()

def calc_epoch_steps(num_files):
    return (num_files + FLAGS.batch_size - 1) // FLAGS.batch_size

def local_swish(x):
    return x * tf.nn.sigmoid(x)

def normalize(input_image, input_mask, dtype):
    input_image = tf.cast(input_image, dtype)
    vgg_means = tf.constant([91.4953, 103.8827, 131.0912])
    input_image -= vgg_means
    input_image /= 255.

    input_mask = tf.cast(input_mask, dtype)
    return input_image, input_mask

@tf.function
def basic_preprocess(filename, input_image, input_mask, dtype):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask, dtype)

    logger.info('image: {}, mask: {}'.format(input_image, input_mask))

    return filename, input_image, input_mask

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

class Unet(tf.keras.Model):
    def __init__(self, output_channels, base_model):
        super(Unet, self).__init__()

        self.base_model = base_model

        self.up_stack = []
        self.final_endpoints = None

        for name, endpoint in self.base_model.endpoints.items():
            logger.info('{}: {}'.format(name, endpoint))

        self.layer_names = ['features', 'reduction_4', 'reduction_3', 'reduction_2', 'reduction_1']

        for name in self.layer_names:
            endpoint = self.base_model.endpoints[name]

            c = endpoint.shape[3]
            up = upsample(c*2, 3)
            self.up_stack.append((name, up))
            logger.info('{}: endpoint: {}, upsampled channels: {}'.format(name, endpoint, c*2))

        self.last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=1, padding='same', activation='softmax')

    def call(self, inputs, training=True):
        x = self.base_model(inputs, training)

        first = True
        for name, up in self.up_stack:
            if not first:
                x = self.base_model.endpoints[name]

            upsampled = up(x)

            if first:
                x = upsampled
            else:
                x = tf.concat([upsampled, endpoint], axis=-1)

            first = True

        x = self.last(x)
        return x

@tf.function
def call_base_model(model, inputs):
    return model(inputs, training=True, features_only=True)

@tf.function
def call_model(model, inputs):
    return model(inputs, training=True)

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]

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

        base_model = efficientnet.build_model(FLAGS.model_name, params)
        if dstrategy is not None:
            dstrategy.experimental_run_v2(call_base_model, args=(base_model, tf.ones((FLAGS.batch_size, image_size, image_size, 3))))

        model = Unet(3, base_model)
        if dstrategy is not None:
            dstrategy.experimental_run_v2(call_model, args=(model, tf.ones((FLAGS.batch_size, image_size, image_size, 3))))
        else:
            dummy_init_unet(model, FLAGS.batch_size, image_size)

        num_params_base = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])
        num_params_unet = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, base model trainable variables/params: {}/{}, unet model trainable variables/params: {}/{}, dtype: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size,
            len(base_model.trainable_variables), int(num_params_base),
            len(model.trainable_variables), int(num_params_unet),
            dtype))

        def create_dataset(name, ann_dir, is_training):
            cards = polygon_dataset.Polygons(ann_dir, logger, image_size, image_size)
            def gen():
                for card in cards.get_cards():
                    orig_im = Image.open(card.image_path)
                    orig_width = orig_im.width
                    orig_height = orig_im.height

                    if orig_im.mode != "RGB":
                        orig_im = orig_im.convert("RGB")

                    size = (image_size, image_size)
                    im = orig_im.resize(size, resample=Image.BILINEAR)

                    img = np.asarray(im)
                    mask = card.card_mask

                    #logger.info('{}: {} -> {}, type: {}, mask: {}, mask non-empty: {}, type: {}'.format(card.image_path, orig_im.size, img.shape, img.dtype, mask.shape, np.any(mask), mask.dtype))

                    yield card.image_path, img, mask


            dataset = tf.data.Dataset.from_generator(gen,
                                                     output_types=(tf.string, tf.uint8, tf.int32),
                                                     output_shapes=(
                                                         tf.TensorShape([]),
                                                         tf.TensorShape([image_size, image_size, 3]),
                                                         tf.TensorShape([image_size, image_size, 1]),
                                                     ))

            dataset = dataset.map(lambda filename, image, mask: basic_preprocess(filename, image, mask, dtype), num_parallel_calls=FLAGS.num_cpus)
            dataset = dataset.cache()
            dataset = dataset.shuffle(FLAGS.batch_size * 2)
            dataset = dataset.batch(FLAGS.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat()

            logger.info('{}: dataset has been created, data dir: {}, images: {}'.format(name, ann_dir, cards.num_images()))

            return dataset, cards.num_images()

        train_dataset, train_num_images = create_dataset('train', FLAGS.train_data_dir, is_training=True)
        eval_dataset, eval_num_images = create_dataset('eval', FLAGS.eval_data_dir, is_training=False)

        def generate_images(tmp_dir):
            data_dir = os.path.join(FLAGS.train_dir, tmp_dir)
            os.makedirs(data_dir, exist_ok=True)

            for filename, image, mask in train_dataset.take(1).unbatch():
                filename = os.path.basename(str(filename))
                image_id = os.path.splitext(filename)[0]

                dst = '{}/{}.png'.format(data_dir, image_id)
                image_draw.draw_im_segm(image.numpy(), [mask.numpy()], dst)

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(step=global_step, optimizer=opt, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        if FLAGS.checkpoint:
            status = checkpoint.restore(FLAGS.checkpoint)
            status.expect_partial()

            logger.info("Restored from external checkpoint {}".format(manager.latest_checkpoint))

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


        cross_entropy_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        def calculate_metrics(logits, labels):
            ce_loss = cross_entropy_loss_object(y_pred=logits, y_true=labels)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=FLAGS.batch_size)

            total_loss = ce_loss
            return total_loss

        def eval_step(images, labels):
            logits = model(images, training=False)
            total_loss = calculate_metrics(logits, labels)

            eval_loss_metric.update_state(total_loss)
            eval_accuracy_metric.update_state(labels, logits)

            return total_loss

        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                total_loss = calculate_metrics(logits, labels)

            variables = model.trainable_variables + base_model.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            opt.apply_gradients(zip(gradients, variables))

            loss_metric.update_state(total_loss)
            accuracy_metric.update_state(labels, logits)

            global_step.assign_add(1)

            return total_loss

        steps_per_epoch = calc_epoch_steps(train_num_images)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_num_images)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch: {}/{}, steps_per_eval: {}/{}'.format(steps_per_epoch, calc_epoch_steps(train_num_images), steps_per_eval, calc_epoch_steps(eval_num_images)))

        @tf.function
        def distributed_step(step_func, images, labels):
            pr_total_losses = dstrategy.experimental_run_v2(step_func, args=(images, labels))

            #total_loss = dstrategy.reduce(tf.distribute.ReduceOp.SUM, pr_total_losses, axis=None)

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


                distributed_step(step_func, images, masks)

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
            _ = run_epoch(train_dataset, train_step, 0)
            eval_steps = run_epoch(eval_dataset, eval_step, steps_per_eval)
            min_metric = eval_accuracy_metric.result()
            logger.info('initial validation metric: {}'.format(min_metric))

        if min_metric < FLAGS.min_eval_metric:
            logger.info('setting minimal evaluation metric {} -> {} from command line arguments'.format(min_metric, FLAGS.min_eval_metric))
            min_metric = FLAGS.min_eval_metric

        for epoch in range(FLAGS.epoch, FLAGS.num_epochs):
            reset_metrics()

            train_steps = run_epoch(train_dataset, train_step, steps_per_epoch)
            eval_steps = run_epoch(eval_dataset, eval_step, steps_per_eval)

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
