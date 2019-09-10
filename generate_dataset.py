import argparse
import cv2
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

import coco
import ssd
import tfrecord_writer

parser = argparse.ArgumentParser()
parser.add_argument('--coco_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--coco_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--num_images', type=int, default=10000000, help='Total number of images to generate')
parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
parser.add_argument('--is_training', action='store_true', help='Training/evaluation augmentation')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def do_work(worker_id, num_images, image_size, anchors_boxes, anchor_areas):
    base = coco.create_coco_iterable(FLAGS.coco_annotations, FLAGS.coco_data_dir, logger)

    coco.complete_initialization(base, image_size, anchors_boxes, anchor_areas, FLAGS.is_training)

    num_images = len(base)
    num_classes = base.num_classes()
    cat_names = base.cat_names()

    writer = tfrecord_writer.tf_records_writer('{}/worker{:02d}'.format(FLAGS.output_dir, worker_id), 0, FLAGS.num_images_per_tfrecord)

    processed = 0
    bboxes = 0

    start_time = time.time()
    for idx in range(num_images):
        idx = idx % num_images

        filename, image_id, image, true_bboxes, true_labels, true_orig_labels = base[idx]

        processed += 1
        bboxes += np.count_nonzero(true_labels)

        image_enc = cv2.imencode('.jpg', image)[1].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _int64_feature(image_id),
            'image': _bytes_feature(bytes(image_enc)),
            'filename': _bytes_feature(bytes(filename, 'utf8')),
            'true_bboxes': _bytes_feature(true_bboxes.tobytes()),
            'true_labels': _bytes_feature(true_labels.tobytes()),
            }))

        data = bytes(example.SerializeToString())
        writer.write(data)

        if idx % 1000 == 0:
            dur = time.time() - start_time
            mean_time = dur / processed
            logger.info('{}: {}: mean bboxes: {:.2f}, mean time: {:.1f} ms'.format(os.getpid(), filename, bboxes / processed, mean_time * 1000))


    writer.close()

def main():
    if FLAGS.logfile:
        handler = logging.FileHandler(FLAGS.logfile, 'a')
        handler.setFormatter(__fmt)
        logger.addHandler(handler)

    model, image_size, anchors_boxes, anchor_areas = ssd.create_model(float, FLAGS.model_name, FLAGS.num_classes)
    num_anchors = anchors_boxes.shape[0]
    logger.info('base_model: {}, num_anchors: {}, image_size: {}'.format(FLAGS.model_name, num_anchors, image_size))

    mp.set_start_method('spawn')

    processes = []
    for i in range(FLAGS.num_cpus):
        p = mp.Process(target=do_work, args=(i, FLAGS.num_images / FLAGS.num_cpus, image_size, anchors_boxes, anchor_areas))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
