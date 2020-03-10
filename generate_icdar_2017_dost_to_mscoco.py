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
parser.add_argument('--dost_dir', type=str, action='append', help='Path to ICDAR 2017 DOST image/gt dir')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def scan_annotations(gt_dir, writer):
    good_exts = ['.jpg', '.png', '.jpeg']

    start_time = time.time()
    processed = 0
    processed_examples = 0

    for gt_fn in os.listdir(gt_dir):
        spl = os.path.splitext(gt_fn)
        if len(spl) != 2:
            continue

        if not spl[1].lower() in good_exts:
            continue

        if spl[0].endswith('_GT'):
            continue

        gt_name = '{}_L.txt'.format(spl[0])

        texts = []
        bboxes = []
        polygons = []

        full_path = os.path.join(gt_dir, gt_name)
        with open(full_path, 'r') as fin:
            for line in fin:
                try:
                    comm = line.split(',')
                    points = comm[:8]
                    text = ','.join(comm[8:]).strip()

                    text.encode('ascii')

                    if text == '###':
                        continue

                    points = np.array(points, dtype=np.float32)
                except:
                    continue

                x = points[..., 0::1]
                y = points[..., 1::1]

                wp = points

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
                polygons.append(wp)

        if len(bboxes) == 0:
            logger.info('skipping {}'.format(spl[0]))
            continue

        try:
            image_filename = os.path.join(gt_dir, gt_fn)
            with open(image_filename, 'rb') as fin:
                image_data = fin.read()
        except:
            continue

        image_id = int(spl[0].split('_')[-1])

        bboxes = np.concatenate(bboxes, 0).astype(np.float32)
        polygons = np.concatenate(polygons, 0).astype(np.float32)
        texts = '<SEP>'.join(texts)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _int64_feature(image_id),
            'image': _bytes_feature(image_data),
            'filename': _bytes_feature(bytes(image_filename, 'UTF-8')),
            'true_bboxes': _bytes_feature(bboxes.tobytes()),
            'word_poly': _bytes_feature(polygons.tobytes()),
            'true_labels': _bytes_feature(bytes(texts, 'UTF-8')),
            }))

        data = bytes(example.SerializeToString())
        writer.write(data)

        processed += 1
        processed_examples += polygons.shape[0]

        if processed % 1000 == 0:
            dur = time.time() - start_time
            mean_time = dur / processed
            logger.info('{}: {}: {}: mean time: {:.1f} ms'.format(
                os.getpid(), processed, image_filename, mean_time * 1000))

    logger.info('saved {} images, examples: {}'.format(processed, processed_examples))

def main():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    writer = tfrecord_writer.tf_records_writer('{}/tfrecord'.format(FLAGS.output_dir), 0, FLAGS.num_images_per_tfrecord)

    for gt_dir in FLAGS.dost_dir:
        scan_annotations(gt_dir, writer)

    writer.close()

if __name__ == '__main__':
    main()
