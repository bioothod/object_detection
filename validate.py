import argparse
import logging
import os
import time

import numpy as np
import multiprocessing as mp
import tensorflow as tf

from PIL import Image

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

def preprocess_image(filename, image_size):
    orig_im = Image.open(filename)
    orig_width = orig_im.width
    orig_height = orig_im.height

    if orig_im.mode != "RGB":
        orig_im = orig_im.convert("RGB")

    size = (image_size, image_size)
    im = orig_im.resize(size, resample=Image.BILINEAR)

    img = np.asarray(im).astype(np.float32)

    #vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
    #img -= vgg_means
    img -= 128.
    img /= 128.

    return filename, img

def run_queue(num_processes, data_source, do_work):
    with mp.Pool(num_processes) as pool:
        for res in pool.imap(func=do_work, iterable=data_source, chunksize=32):
            yield res

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


    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]
    logger.info('starting with model: {}, image_size: {}'.format(FLAGS.model_name, image_size))

    dtype = tf.float32

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

    params = {
        'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
        'data_format': FLAGS.data_format,
        'relu_fn': tf.nn.swish
    }

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

    output_channels = 3
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')

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
