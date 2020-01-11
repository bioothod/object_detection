import argparse
import cv2
import json
import logging
import os
import sys
import time

import multiprocessing as mp
import numpy as np
import tensorflow as tf

logger = logging.getLogger('generate')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

import tfrecord_writer

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, action='append', help='Path to ICDAR 2017 ground truth dir')
parser.add_argument('--image_dir', type=str, action='append', help='Path to ICDAR 2017 image dir')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def scan_dirs(dirnames):
    good_exts = ['.jpg', '.png', '.jpeg']
    filenames = {}

    for dirname in dirnames:
        logger.info('{}: start scanning'.format(dirname))
        for fn in os.listdir(dirname):
            full = os.path.join(dirname, fn)

            if os.path.isdir(full):
                other = scan_dir(full)
                filenames.update(other)
                continue

            spl = os.path.splitext(fn.lower())
            if len(spl) != 2:
                continue

            if not spl[1] in good_exts:
                continue

            filenames[spl[0]] = full

    logger.info('{}: images: {}'.format(dirname, len(filenames)))
    return filenames

def scan_annotations(gt_dir, writer, image_fns):
    start_time = time.time()
    processed = 0

    for gt_fn in os.listdir(gt_dir):
        gt_id = gt_fn[3:-4]
        image_id = int(gt_id[4:])


        texts = []
        bboxes = []

        full_path = os.path.join(gt_dir, gt_fn)
        with open(full_path, 'r') as fin:
            for line in fin:
                comm = line.split(',')
                points = comm[:8]
                lang = comm[8]
                text = ','.join(comm[9:])

                if lang != 'Latin':
                    continue

                text = [t.strip() for t in text.split()]
                text = ''.join(text)

                points = np.array(points, dtype=np.float32)

                x = points[..., 0::2]
                y = points[..., 1::2]
                xmin = x.min()
                xmax = x.max()
                ymin = y.min()
                ymax = y.max()

                cx = (xmax + xmin) / 2
                cy = (ymax + ymin) / 2
                h = ymax - ymin
                w = xmax - xmin

                bb = np.array([cx, cy, h, w], dtype=np.float32)

                bboxes.append(bb)
                texts.append(text)

        if len(bboxes) == 0:
            continue

        try:
            image_filename = image_fns[gt_id]
            with open(image_filename, 'rb') as fin:
                image_data = fin.read()
        except:
            continue

        bboxes = np.concatenate(bboxes, 0).astype(np.float32)
        texts = '<SEP>'.join(texts)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _int64_feature(image_id),
            'image': _bytes_feature(image_data),
            'filename': _bytes_feature(bytes(image_filename, 'UTF-8')),
            'true_bboxes': _bytes_feature(bboxes.tobytes()),
            'true_labels': _bytes_feature(bytes(texts, 'UTF-8')),
            }))

        data = bytes(example.SerializeToString())
        writer.write(data)

        processed += 1

        if processed % 1000 == 0:
            dur = time.time() - start_time
            mean_time = dur / processed
            logger.info('{}: {}: {}: mean time: {:.1f} ms'.format(
                os.getpid(), processed, image_filename, mean_time * 1000))

    logger.info('saved {} images'.format(processed))

def main():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    writer = tfrecord_writer.tf_records_writer('{}/tfrecord'.format(FLAGS.output_dir), 0, FLAGS.num_images_per_tfrecord)

    image_fns = scan_dirs(FLAGS.image_dir)

    for gt_dir in FLAGS.gt_dir:
        scan_annotations(gt_dir, writer, image_fns)

    writer.close()

if __name__ == '__main__':
    main()
