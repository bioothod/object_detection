import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

import batch
import efficientnet
import preprocess
import validation

from tensorflow.keras import Model
import tensorflow.keras.layers as layers

logger = logging.getLogger('vggface_emotions')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, required=True, help='Json file which describes classes and contains lists of filenames of data files')
parser.add_argument('--validation_json', type=str, required=True, help='Validation json file which describes classes and contains lists of filenames of data files')
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

def tf_read_image(filename, label, image_size, is_training, dtype):
    image = tf.io.read_file(filename)

    #image = tf.image.decode_jpeg(image, dct_method='INTEGER_ACCURATE', channels=3)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.processing_function(image, image_size, image_size, is_training, dtype, FLAGS.autoaugment_name)

    return image, label

def calc_epoch_steps(bg):
    return (bg.num_files() + FLAGS.batch_size - 1) // FLAGS.batch_size

def local_swish(x):
    return x * tf.nn.sigmoid(x)

def train():
    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    train_bg = batch.generator(FLAGS.input_json, split_to=1, use_chunk=0)
    eval_bg = batch.generator(FLAGS.validation_json, 1, 0)

    num_classes = train_bg.num_classes()
    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]

    num_replicas = 1
    dstrategy = tf.distribute.MirroredStrategy()
    num_replicas = dstrategy.num_replicas_in_sync
    with dstrategy.scope():
    #if True:
        initial_learning_rate = FLAGS.initial_learning_rate * num_replicas
        FLAGS.batch_size *= num_replicas

        params = {
            'num_classes': num_classes,
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

        #tf.keras.backend.set_learning_phase(1)

        global_step = tf.Variable(1, dtype=tf.int64, name='global_step')
        model = efficientnet.build_model(model_name=FLAGS.model_name, override_params=params)
        for name, endpoint in model.endpoints:
            logger.info('{}: {}'.format(name, endpoint))

        exit(0)

        num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

        logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, num_classes: {}, trainable variables: {}, trainable params: {}, dtype: {}, autoaugment_name: {}'.format(
            num_replicas, checkpoint_dir, FLAGS.model_name, image_size, num_classes, len(model.trainable_variables), int(num_params), dtype, FLAGS.autoaugment_name))

        has_moving_average_decay = (FLAGS.moving_average_decay > 0)
        # This is essential, if using a keras-derived model.

        restore_vars_dict = None

        if has_moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(decay=FLAGS.moving_average_decay, num_updates=global_step)

            ema_vars = tf.trainable_variables() + tf.compat.v1.get_collection('moving_vars')
            for v in tf.global_variables():
                # We maintain mva for batch norm moving mean and variance as well.
                if 'moving_mean' in v.name or 'moving_variance' in v.name:
                    ema_vars.append(v)

            ema_vars = list(set(ema_vars))

        def create_dataset(name, bg, is_training):
            paths, labels = bg.get()

            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            dataset = dataset.map(lambda path, label: tf_read_image(path, label, image_size, is_training, dtype), num_parallel_calls=FLAGS.num_cpus)
            #dataset = dataset.cache()
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(FLAGS.batch_size * 2)
            dataset = dataset.batch(FLAGS.batch_size)
            dataset = dataset.repeat()

            logger.info('{}: dataset has been created'.format(name))

            return dataset

        train_dataset = create_dataset('train', train_bg, is_training=True)
        eval_dataset = create_dataset('eval', eval_bg, is_training=False)

        opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        callbacks_list = []

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.2, patience=10, min_lr=1e-5, mode='max', verbose=1, cooldown=2)
        callbacks_list.append(lr_callback)

        filepath = os.path.join(checkpoint_dir, "model.ckpt-{epoch:02d}-{val_sparse_categorical_accuracy:.4f}.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list.append(checkpoint)

        steps_per_epoch = calc_epoch_steps(train_bg)
        if FLAGS.steps_per_epoch > 0:
            steps_per_epoch = FLAGS.steps_per_epoch
        steps_per_eval = calc_epoch_steps(eval_bg)
        if FLAGS.steps_per_eval > 0:
            steps_per_eval = FLAGS.steps_per_eval

        logger.info('steps_per_epoch: {}/{}, steps_per_eval: {}/{}'.format(steps_per_epoch, calc_epoch_steps(train_bg), steps_per_eval, calc_epoch_steps(eval_bg)))

        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        if FLAGS.checkpoint:
            model.fit(train_dataset, epochs=0, steps_per_epoch=1)
            logger.info('loading weights from {}'.format(FLAGS.checkpoint))
            model.load_weights(FLAGS.checkpoint)

        model.fit(train_dataset,
                epochs=FLAGS.num_epochs, initial_epoch=FLAGS.epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks_list, verbose=1,
                validation_data=eval_dataset, validation_steps=steps_per_eval)

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
