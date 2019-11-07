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
import image as image_draw
import tfrecord_writer

parser = argparse.ArgumentParser()
parser.add_argument('--coco_text_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--coco_train_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--coco_eval_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--image_size', type=int, default=1024, help='Image size')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--num_images', type=int, default=10000000, help='Total number of images to generate')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
parser.add_argument('--do_augmentation', action='store_true', help='Whether to store original images or augmented')
parser.add_argument('--is_training', action='store_true', help='Training/evaluation augmentation')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def do_work(worker_id, step, num_images, image_size):
    base = coco.COCOText(FLAGS.coco_text_annotations, FLAGS.coco_train_data_dir, FLAGS.coco_eval_data_dir, logger)

    if num_images == 0:
        num_images = base.num_images(FLAGS.is_training)

    images_dir = os.path.join(FLAGS.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    writer = tfrecord_writer.tf_records_writer('{}/worker{:02d}'.format(FLAGS.output_dir, worker_id), 0, FLAGS.num_images_per_tfrecord)

    processed = 0
    bboxes = 0
    crops_saved = 0

    start_time = time.time()
    for idx in range(worker_id, num_images, step):
        idx = idx % base.num_images(FLAGS.is_training)

        try:
            if FLAGS.do_augmentation:
                filename, image_id, image, true_bboxes, true_labels = base.process_image(idx, FLAGS.is_training, coco.get_text_train_augmentation(image_size))
            else:
                filename, image_id, image, true_bboxes, true_labels = base.process_image(idx, FLAGS.is_training, None)
        except coco.ProcessingError as e:
            continue

        processed += 1
        bboxes += true_bboxes.shape[0]

        image = image.astype(np.uint8)

        for bb, text in zip(true_bboxes, true_labels):
            if text == '<SKIP>':
                continue
            if text == ' ':
                continue

            cx, cy, h, w = bb

            # drop vertical texts
            if h > 2*w:
                continue

            h *= 1.1
            w *= 1.1

            x0 = max(int(cx - w/2), 0)
            x1 = min(int(cx + w/2), image.shape[1]-1)
            y0 = max(int(cy - h/2), 0)
            y1 = min(int(cy + h/2), image.shape[0]-1)

            bb = [x0, y0, x1, y1]

            img = image[y0:y1+1, x0:x1+1, :]

            #logger.info('{}: bb: {}, img: {}, text: {}'.format(filename, bb, img.shape, text))
            image_enc = cv2.imencode('.png', img)[1].tostring()

            if crops_saved <= 10:
                new_anns = [(bb, None, None)]

                dst = '{}/{:02d}_{}_{}.png'.format(images_dir, worker_id, image_id, crops_saved)
                logger.info('{}: true anchors: {}'.format(dst, new_anns))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_draw.draw_im(img, new_anns, dst, {})


            example = tf.train.Example(features=tf.train.Features(feature={
                'image_id': _int64_feature(image_id),
                'image': _bytes_feature(bytes(image_enc)),
                'filename': _bytes_feature(bytes(filename, 'utf8')),
                'text': _bytes_feature(bytes(text, 'utf8')),
                }))

            data = bytes(example.SerializeToString())
            writer.write(data)
            crops_saved += 1

        if idx % 1000 == 0:
            dur = time.time() - start_time
            mean_time = dur / processed
            logger.info('{}: {}: images: {}: saved crops: {}, mean time: {:.1f} ms'.format(
                os.getpid(), filename, processed, crops_saved, mean_time * 1000))


    writer.close()

def main():
    if FLAGS.logfile:
        handler = logging.FileHandler(FLAGS.logfile, 'a')
        handler.setFormatter(__fmt)
        logger.addHandler(handler)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    logger.info('image_size: {}'.format(FLAGS.image_size))

    mp.set_start_method('spawn')

    processes = []
    for i in range(FLAGS.num_cpus):
        num = FLAGS.num_images

        p = mp.Process(target=do_work, args=(i, FLAGS.num_cpus, num, FLAGS.image_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
