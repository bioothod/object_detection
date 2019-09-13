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
import efficientnet
import image as image_draw
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

def do_work(worker_id, num_images, image_size):
    base = coco.create_coco_iterable(FLAGS.coco_annotations, FLAGS.coco_data_dir, logger)

    coco.complete_initialization(base, image_size, [], [], FLAGS.is_training)

    num_images = len(base)
    num_classes = base.num_classes()
    cat_names = base.cat_names()

    images_dir = os.path.join(FLAGS.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    writer = tfrecord_writer.tf_records_writer('{}/worker{:02d}'.format(FLAGS.output_dir, worker_id), 0, FLAGS.num_images_per_tfrecord)

    processed = 0
    bboxes = 0

    start_time = time.time()
    for idx in range(num_images):
        idx = idx % num_images

        try:
            filename, image_id, image, true_bboxes, true_labels = base.process_image(idx, base.train_augmentation, return_orig_format=True)
        except coco.ProcessingError as e:
            continue

        processed += 1
        bboxes += np.count_nonzero(true_labels)

        image = image * 128 + 128
        image = image.astype(np.uint8)
        image_enc = cv2.imencode('.png', image)[1].tostring()

        if processed <= 10:
            new_anns = []
            for bb, cat_id in zip(true_bboxes, true_labels):
                cx, cy, h, w = bb
                x0 = cx - w/2
                x1 = cx + w/2
                y0 = cy - h/2
                y1 = cy + h/2

                bb = [x0, y0, x1, y1]
                new_anns.append((bb, cat_id))

            dst = '{}/{:02d}_{}_{}.png'.format(images_dir, worker_id, image_id, processed)
            logger.info('{}: true anchors: {}'.format(dst, new_anns))

            image_draw.draw_im(image, new_anns, dst, {})


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
            logger.info('{}: {}: mean bboxes: {:.2f}, mean time: {:.1f} ms, bboxes.dtype: {}, labels.dtype: {}'.format(
                os.getpid(), filename, bboxes / processed, mean_time * 1000, true_bboxes.dtype, true_labels.dtype))


    writer.close()

def main():
    if FLAGS.logfile:
        handler = logging.FileHandler(FLAGS.logfile, 'a')
        handler.setFormatter(__fmt)
        logger.addHandler(handler)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    image_size = efficientnet.efficientnet_params(FLAGS.model_name)[2]
    logger.info('base_model: {}, image_size: {}'.format(FLAGS.model_name, image_size))

    mp.set_start_method('spawn')

    processes = []
    for i in range(FLAGS.num_cpus):
        p = mp.Process(target=do_work, args=(i, FLAGS.num_images / FLAGS.num_cpus, image_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
