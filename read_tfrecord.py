import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

import encoder
import preprocess

logger = logging.getLogger('reader')

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=10, help='Number of records to extract.')
parser.add_argument('--skip', type=int, default=0, help='Number of records to skip.')
parser.add_argument('--verbose', action='store_true', help='Specifies whether to print per-file information.')
parser.add_argument('--extract', type=str, required=True, help='Directory to store extracted images.')
parser.add_argument('--crop_size', type=str, default='8x64', help='Sampling crop size, HxW')
parser.add_argument('--logfile', type=str, help='Logfile')
parser.add_argument('filenames', nargs='*', type=str, help='Training TFRecord filename')
FLAGS = parser.parse_args()

def tf_parse_example(base_dir, serialized_example, crop_size):
    features = tf.io.parse_single_example(serialized_example,
        features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'true_labels': tf.io.FixedLenFeature([], tf.string),
            'word_poly': tf.io.FixedLenFeature([], tf.string),
        })

    filename = features['filename']

    image = features['image']
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    word_poly = tf.io.decode_raw(features['word_poly'], tf.float32)
    word_poly = tf.reshape(word_poly, [-1, 4, 2])


    if False:
        orig_image_height = tf.cast(tf.shape(image)[0], tf.float32)
        orig_image_width = tf.cast(tf.shape(image)[1], tf.float32)

        mx = tf.maximum(orig_image_height, orig_image_width)
        mx_int = tf.cast(mx, tf.int32)
        image = tf.image.pad_to_bounding_box(image,
                    tf.cast((mx - orig_image_height) / 2., tf.int32),
                    tf.cast((mx - orig_image_width) / 2., tf.int32),
                    mx_int,
                    mx_int)

        xdiff = (mx - orig_image_width) / 2
        ydiff = (mx - orig_image_height) / 2

        add = tf.stack([xdiff, ydiff])
        word_poly += add

        image_size = 800

        current_image_size = tf.cast(tf.shape(image)[1], tf.float32)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, dtype)

        word_poly = word_poly / current_image_size * image_size
        word_poly = tf.cast(word_poly, dtype)

    text_labels = tf.strings.split(features['true_labels'], '<SEP>')

    selected_features = tf.TensorArray(image.dtype, size=0, dynamic_size=True)
    written = 0
    selected_features, written = encoder.sample_crops_for_single_image(image, word_poly, crop_size, selected_features, written)

    for idx in tf.range(selected_features.size()):
        sf = selected_features.gather(idx)
        sf = tf.squeeze(sf)
        sf = tf.cast(sf, tf.uint8)

        enc = tf.io.encode_jpeg(sf, format='rgb', quality=100)

        label = text_labels[idx]

        def fmt(filename, label):
            filename = str(filename.numpy(), 'utf8')
            label = str(label.numpy(), 'utf8')

            filename = os.path.basename(filename)
            filename = os.path.splitext(filename)[0]

            return '{}/{}_{}.jpg'.format(base_dir, filename, label)

        new_fn = tf.py_function(fmt, [filename, label], tf.string)
        tf.io.write_file(new_fn, enc)

    return filename, selected_features.size()

def main():
    crop_size = [int(c) for c in FLAGS.crop_size.split('x')[:2]]

    if FLAGS.extract:
        os.makedirs(FLAGS.extract, exist_ok=True)

    def scan_dirs(dirname):
        filenames = []
        for fn in os.listdir(dirname):
            fn = os.path.join(dirname, fn)
            if os.path.isfile(fn):
                filenames.append(fn)
            elif os.path.isdir(fn) or os.path.islink(fn):
                filenames += scan_dirs(fn)

        return filenames

    filenames = []
    for filename in FLAGS.filenames:
        if os.path.isfile(filename):
            filenames.append(filename)
        else:
            filenames += scan_dirs(filename)

    total_filenames = len(filenames)
    logger.info('input filenames pattern: {}, files found: {}'.format(FLAGS.filenames, total_filenames))

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.skip(FLAGS.skip)
    dataset = dataset.take(FLAGS.num)
    dataset = dataset.map(lambda img: tf_parse_example(FLAGS.extract, img, crop_size), num_parallel_calls=32)

    total_crops = 0
    total_images = 0
    try:
        for filename, num_crops in dataset:
            filename = str(filename.numpy(), 'utf8')
            num_crops = num_crops.numpy()

            total_images += 1
            total_crops += num_crops

            if total_images % 100 == 0:
                logger.info("images: {}, crops: {}".format(total_images, total_crops))

            if FLAGS.verbose:
                logger.info("filename: {}, crops: {}, total_images: {}, total_crops: {}".format(filename, num_crops, total_images, total_crops))

    except tf.errors.OutOfRangeError as e:
        logger.error('out of range exception: {}'.format(e))
    #except Exception as e:
    #    print('exception: {}'.format(e))
    #    pass

    logger.info("images: {}, crops: {}".format(total_images, total_crops))

if __name__ == '__main__':
    logger.propagate = False
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    if FLAGS.logfile:
        handler = logging.FileHandler(FLAGS.logfile, 'a')
        handler.setFormatter(__fmt)
        logger.addHandler(handler)

    main()
