import argparse
import logging
import os
import time

import numpy as np
import tensorflow as tf

import efficientnet
import image as image_draw
import preprocess
import unet

logger = logging.getLogger('segmentation')
logger.propagate = False

def generate_images(filenames, images, masks, data_dir):
    vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)

    for filename, image, mask in zip(filenames, images, masks):
        filename = str(filename)
        filename = os.path.basename(filename)
        image_id = os.path.splitext(filename)[0]

        image = image.numpy() * 128. + 128
        image = image.astype(np.uint8)
        logger.info('{}: min: {}, max: {}'.format(filename, np.min(image), np.max(image)))

        dst = '{}/{}.png'.format(data_dir, image_id)
        image_draw.draw_im_segm(image, [mask.numpy()], dst)

@tf.function
def basic_preprocess(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.processing_function(image, image_size, image_size, False, dtype)

    return filename, image

@tf.function
def load_image(datapoint, image_size, is_training, dtype):
    image = datapoint['image']
    mask = datapoint['segmentation_mask']
    filename = datapoint['file_name']

    image = preprocess.processing_function(image, image_size, image_size, is_training, dtype)
    mask = preprocess.try_resize(mask, image_size, image_size) - 1

    if tf.random.uniform(()) > 0.5 and is_training:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    return filename, image, mask

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory where to store masked images')
    parser.add_argument('--dataset', type=str, choices=['card_images', 'oxford_pets'], default='images', help='Dataset type')
    parser.add_argument('--num_cpus', type=int, default=32, help='Number of preprocessing processes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Load model weights from this file')
    parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
    parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
    FLAGS = parser.parse_args()

    dtype = tf.float32

    params = {
        'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
        'data_format': FLAGS.data_format,
        'relu_fn': tf.nn.swish
    }

    num_classes = 3
    base_model, model, image_size = unet.create_model(params, dtype, FLAGS.model_name, num_classes)

    if FLAGS.dataset == 'card_images':
        filenames = [os.path.join(FLAGS.data_dir, fn) for fn in os.listdir(FLAGS.data_dir) if os.path.splitext(fn.lower())[-1] in ['.png', '.jpg', '.jpeg']]

        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(lambda filename: basic_preprocess(filename, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)
    elif FLAGS.dataset == 'oxford_pets':
        import tensorflow_datasets as tfds

        dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True, data_dir=FLAGS.data_dir)

        dataset = dataset['test'].map(lambda datapoint: load_image(datapoint, image_size, False, dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)


    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint)
    status.expect_partial()
    logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))

    @tf.function
    def eval_step(images):
        logits = model(images, training=False)
        masks = tf.argmax(logits, axis=-1)
        return masks

    os.makedirs(FLAGS.dst_dir, exist_ok=True)

    start_time = time.time()
    total_images = 0
    for t in dataset:
        filenames = t[0]
        images = t[1]

        masks = eval_step(images)
        total_images += len(filenames)

        generate_images(filenames, images, masks, FLAGS.dst_dir)
        logger.info('saved {}/{} images'.format(len(filenames), total_images))

    dur = time.time() - start_time
    logger.info('processed {} images, time: {:.2f} seconds, {:.1f} ms per image'.format(total_images, dur, dur / total_images * 1000.))
