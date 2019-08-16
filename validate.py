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
        filename = os.path.basename(str(filename))
        image_id = os.path.splitext(filename)[0]

        image = image.numpy() * 255. + vgg_means
        image = image.astype(np.uint8)
        logger.info('{}: min: {}, max: {}'.format(str(filename), np.min(image), np.max(image)))

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

    vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
    img -= vgg_means
    img /= 255.

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

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory where to store masked images')
    parser.add_argument('--num_cpus', type=int, default=32, help='Number of preprocessing processes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Load model weights from this file')
    parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
    parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
    FLAGS = parser.parse_args()


    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]
    logger.info('starting with model: {}, image_size: {}'.format(FLAGS.model_name, image_size))

    filenames = [os.path.join(FLAGS.data_dir, fn) for fn in os.listdir(FLAGS.data_dir) if os.path.splitext(fn.lower())[-1] in ['.png', '.jpg', '.jpeg']]
    dtype = tf.float32

    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(lambda filename: basic_preprocess(filename, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)


    params = {
        'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
        'data_format': FLAGS.data_format,
        'relu_fn': tf.nn.swish
    }
    base_model = efficientnet.build_model(FLAGS.model_name, params)
    dummy_img = tf.zeros((FLAGS.batch_size, image_size, image_size, 3))

    @tf.function
    def dummy_call():
        base_model(dummy_img, training=False)
    dummy_call()
    model = unet.Unet(3, base_model)

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint)
    logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))

    @tf.function
    def eval_step(images):
        logits = model(images, training=False)
        masks = tf.argmax(logits, axis=-1)
        return masks

    os.makedirs(FLAGS.dst_dir, exist_ok=True)

    start_time = time.time()
    total_images = 0
    for filenames, images in dataset:
        masks = eval_step(images)
        total_images += len(filenames)

        generate_images(filenames, images, masks, FLAGS.dst_dir)
        logger.info('saved {}/{} images'.format(len(filenames), total_images))

    dur = time.time() - start_time
    logger.info('processed {} images, time: {:.2f} seconds, {:.1f} ms per image'.format(total_images, dur, dur / total_images * 1000.))
