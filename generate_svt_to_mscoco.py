import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET

import multiprocessing as mp
import numpy as np
import tensorflow as tf

from shapely.geometry import Polygon

logger = logging.getLogger('generate')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

import tfrecord_writer

parser = argparse.ArgumentParser()
parser.add_argument('--svt_annotations', type=str, help='Path to SVT XML annotation')
parser.add_argument('--svt_dir', type=str, help='Path to SVT root dir, which contains image dir')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def scan_annotations(xml_ann, writer):
    logger.info('opening {}'.format(xml_ann))
    tree = ET.parse(xml_ann)
    root = tree.getroot()

    logger.info('xml has been opened, starting ')
    start_time = time.time()
    processed = 0

    for image_node in root:
        image_name = image_node.find('imageName').text
        image_id = 0

        texts = []
        bboxes = []
        polygons = []

        text_nodes = image_node.find('taggedRectangles')
        for tn in text_nodes:
            x0 = tn.attrib['x']
            y0 = tn.attrib['y']
            h = tn.attrib['height']
            w = tn.attrib['width']

            text = tn.find('tag').text

            text = [t.strip() for t in text.split()]
            text = ''.join(text)

            points = [[x0, y0], [x0+w, y0], [x0+w, y0+h], [x0, y0+h]]
            points = np.array(points, dtype=np.float32)
            x = points[..., 0::1]
            y = points[..., 1::1]

            if points.shape == (4, 2):
                wp = points
            else:
                polygon = Polygon(points)
                wp_xy = polygon.minimum_rotated_rectangle.exterior.coords.xy
                wp = np.array([[x, y] for x, y in zip(*wp_xy)])[1:, :]

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
            continue

        image_filename = os.path.join(FLAGS.svt_dir, image_name)
        with open(image_filename, 'rb') as fin:
            image_data = fin.read()

        bboxes = np.concatenate(bboxes, 0).astype(np.float32)
        polygons = np.concatenate(polygons, 0).astype(np.float32)
        texts = '<SEP>'.join(texts)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _int64_feature(image_id),
            'image': _bytes_feature(image_data),
            'filename': _bytes_feature(bytes(image_filename, 'UTF-8')),
            'true_bboxes': _bytes_feature(bboxes.tobytes()),
            'true_labels': _bytes_feature(bytes(texts, 'UTF-8')),
            'word_poly': _bytes_feature(polygons.tobytes()),
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

    scan_annotations(FLAGS.svt_annotations, writer)

    writer.close()

if __name__ == '__main__':
    main()
