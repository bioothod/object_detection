import argparse
import cv2
import logging
import os
import sys
import time

import multiprocessing as mp
import numpy as np
import tensorflow as tf
import scipy.io as sio

logger = logging.getLogger('generate')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

import image as image_draw
import tfrecord_writer
import synth_text

parser = argparse.ArgumentParser()
parser.add_argument('--synth_text_annotations', type=str, default='/shared2/object_detection/datasets/text/synth_text/SynthText/gt.mat', help='Path to SynthText dataset: annotations matlab file')
parser.add_argument('--synt_text_data_dir', type=str, default='/shared2/object_detection/datasets/text/synth_text/SynthText/', help='Path to SynthText dataset: image directory')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--num_images', type=int, default=0, help='Total number of images to generate')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
parser.add_argument('--do_augmentation', action='store_true', help='Whether to store original images or augmented')
parser.add_argument('--is_training', action='store_true', help='Training/evaluation augmentation')
FLAGS = parser.parse_args()

logger = logging.getLogger('detection')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def do_work(worker_id, tup):
    filenames, char_polys, word_polys, texts = tup

    logger.info('{}: started processing {} examples'.format(worker_id, len(filenames)))

    if not FLAGS.is_training or FLAGS.num_images == 0:
        num_images = len(filenames)

    images_dir = os.path.join(FLAGS.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    writer = tfrecord_writer.tf_records_writer('{}/worker{:02d}'.format(FLAGS.output_dir, worker_id), 0, FLAGS.num_images_per_tfrecord)

    processed = 0

    start_time = time.time()
    for filename, char_poly, word_poly, text in zip(filenames, char_polys, word_polys, texts):
        filename = filename[0]

        image_filename = os.path.join(FLAGS.synt_text_data_dir, filename)
        with open(image_filename, 'rb') as fin:
            image_data = fin.read()

        if len(word_poly.shape) == 2:
            word_poly = np.expand_dims(word_poly, 2)
        if len(char_poly.shape) == 2:
            char_poly = np.expand_dims(char_poly, 2)

        word_poly = np.transpose(word_poly, [2, 1, 0])
        char_poly = np.transpose(char_poly, [2, 1, 0])

        if processed < 0:
            new_anns = []
            for wp in word_poly:
                new_anns.append((None, wp, None))

            image = cv2.imread(image_filename)
            fn = os.path.basename(filename)
            dst = os.path.join(images_dir, fn)
            image_draw.draw_im(image, new_anns, dst, {})

        texts = []
        for t in text:
            s = [w.strip() for w in t.split()]
            texts += s

        text_concat = ''.join(texts)
        texts = '<SEP>'.join(texts)

        char_poly = char_poly.astype(np.float32)
        word_poly = word_poly.astype(np.float32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_data),
            'filename': _bytes_feature(bytes(filename, 'UTF-8')),
            'char_poly': _bytes_feature(char_poly.tobytes()),
            'word_poly': _bytes_feature(word_poly.tobytes()),
            'text': _bytes_feature(bytes(texts, 'UTF-8')),
            'text_concat': _bytes_feature(bytes(text_concat, 'UTF-8')),
            }))

        data = bytes(example.SerializeToString())
        writer.write(data)

        processed += 1

        if processed % 1000 == 0:
            dur = time.time() - start_time
            mean_time = dur / processed
            logger.info('{}: {}: mean time: {:.1f} ms'.format(
                os.getpid(), filename, mean_time * 1000))


    writer.close()

def main():
    if FLAGS.logfile:
        handler = logging.FileHandler(FLAGS.logfile, 'a')
        handler.setFormatter(__fmt)
        logger.addHandler(handler)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    filenames, char_polys, word_polys, texts = synth_text.load_synth_dataset(FLAGS.synth_text_annotations, FLAGS.synt_text_data_dir)
    logger.info('opened {}, splitting {} examples into {} groups'.format(FLAGS.synth_text_annotations, len(filenames), FLAGS.num_cpus))

    filenames = np.array_split(filenames, FLAGS.num_cpus)
    char_polys = np.array_split(char_polys, FLAGS.num_cpus)
    word_polys = np.array_split(word_polys, FLAGS.num_cpus)
    texts = np.array_split(texts, FLAGS.num_cpus)

    mp.set_start_method('spawn')

    processes = []

    for idx, (fn, cp, wp, t) in enumerate(zip(filenames, char_polys, word_polys, texts)):
        tup = (fn, cp, wp, t)

        p = mp.Process(target=do_work, args=(idx, tup))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
