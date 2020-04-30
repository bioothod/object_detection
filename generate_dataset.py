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

from collections import defaultdict, OrderedDict

import image as image_draw
import tfrecord_writer

parser = argparse.ArgumentParser()
parser.add_argument('--coco_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--coco_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--num_images_per_tfrecord', type=int, default=10000, help='Number of images in single tfsecord')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tfrecords')
parser.add_argument('--logfile', type=str, help='Logfile')
FLAGS = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class COCO:
    def __init__(self, annotations_json, data_dir):
        start_time = time.time()

        with open(annotations_json, 'r') as fin:
            self.dataset = json.load(fin)

        self.data_dir = data_dir

        self.image_filenames = []
        self.image_ids = []

        self.anns = {}
        self.imgs = {}
        self.cats = {}

        self.img2anns = defaultdict(list)
        self.coco_cat2imgs = defaultdict(list)

        self.coco2flat_cat = {}

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                image_id = ann['image_id']

                if ann.get('iscrowd', 0) != 0:
                    continue

                self.img2anns[image_id].append(ann)
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                image_id = img['id']

                self.imgs[image_id] = img

                self.image_ids.append(image_id)
                self.image_filenames.append(os.path.join(self.data_dir, img['file_name']))

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                image_id = ann['image_id']

                mscoco_category_id = ann['category_id']
                cat_name = self.cats[mscoco_category_id]['name']

                self.coco_cat2imgs[mscoco_category_id].append(image_id)

            cats_sorted = OrderedDict(sorted(self.cats.items()))
            for category_id, cat in cats_sorted.items():
                self.coco2flat_cat[category_id] = len(self.coco2flat_cat)

        logger.info('Loaded MS-COCO dataset: {:.2f} sec'.format(time.time() - start_time))
        for ms_cat_id, flat_cat_id in self.coco2flat_cat.items():
            cat = self.cats[ms_cat_id]
            cat_name = cat['name']

            images = self.coco_cat2imgs[ms_cat_id]
            num_anns = 0
            for image_id in images:
                anns = self.img2anns[image_id]
                num_anns += len(anns)

            logger.info('{}: ms_coco_id: {}, flat_id: {}, images: {}, annotations: {}'.format(
                cat_name, ms_cat_id, flat_cat_id, len(images), num_anns))

        logger.info('images: {}, annotations: {}, categories: {}'.format(len(self.imgs), len(self.anns), len(self.cats)))

    def save_flat_category_ids(self, filename):
        d = {}
        for ms_cat_id, flat_cat_id in self.coco2flat_cat.items():
            cat = self.cats[ms_cat_id]
            cat_name = cat['name']
            d[cat_name] = flat_cat_id

        with open(filename, 'w') as fout:
            json.dump(d, fout, indent=2)

        logger.info('name to flat category ids has been stored to {}'.format(filename))

    def flat_ids_to_names(self):
        cat_names = {}
        for ms_cat_id, flat_cat_id in self.coco2flat_cat.items():
            cat = self.cats[ms_cat_id]
            cat_name = cat['name']
            cat_names[flat_cat_id] = cat_name

        return cat_names

def do_work(worker_id, step, coco):
    images_dir = os.path.join(FLAGS.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    writer = tfrecord_writer.tf_records_writer('{}/worker{:02d}'.format(FLAGS.output_dir, worker_id), 0, FLAGS.num_images_per_tfrecord)

    total_images = 0
    total_bboxes = 0

    image_filenames = np.array_split(coco.image_filenames, step)[worker_id]
    image_ids = np.array_split(coco.image_ids, step)[worker_id]

    start_time = time.time()
    def log_progress():
        dur = time.time() - start_time
        mean_time = dur / total_images
        logger.info('{}: {}: mean bboxes: {:.2f}, mean time: {:.1f} ms per image'.format(
            os.getpid(), filename, total_bboxes / total_images, mean_time * 1000))

    for image_id, filename in zip(image_ids, image_filenames):

        anns = coco.img2anns[image_id]
        bboxes = []
        labels = []
        for ann in anns:
            # format: x0, y0, w, h

            bb = ann['bbox']
            ms_cat_id = ann['category_id']
            flat_cat_id = coco.coco2flat_cat[ms_cat_id]

            bboxes.append(bb)
            labels.append(flat_cat_id)

        total_images += 1
        total_bboxes += len(bboxes)

        with open(filename, 'rb') as fin:
            image_data = fin.read()

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        if total_images <= 10:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            new_anns = []
            for bb, cat_id in zip(bboxes, labels):
                x0, y0, w, h = bb
                x1 = x0 + w
                y1 = y0 + h

                bb = [x0, y0, x1, y1]
                new_anns.append((bb, None, cat_id))

            dst = '{}/{:02d}_{}_{}.png'.format(images_dir, worker_id, image_id, total_images)
            logger.info('{}: true anchors: {}'.format(dst, new_anns))

            image_draw.draw_im(image, new_anns, dst, coco.flat_ids_to_names())


        example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _int64_feature(image_id),
            'image': _bytes_feature(bytes(image_data)),
            'filename': _bytes_feature(bytes(filename, 'utf8')),
            'true_bboxes': _bytes_feature(bboxes.tobytes()),
            'true_labels': _bytes_feature(labels.tobytes()),
            }))

        data = bytes(example.SerializeToString())
        writer.write(data)

        if total_images % 1000 == 0:
            log_progress()

    log_progress()

    writer.close()

def main():
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

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    coco = COCO(FLAGS.coco_annotations, FLAGS.coco_data_dir)
    coco.save_flat_category_ids(os.path.join(FLAGS.output_dir, 'cat2flat_id.json'))

    mp.set_start_method('spawn')

    processes = []
    for i in range(FLAGS.num_cpus):
        p = mp.Process(target=do_work, args=(i, FLAGS.num_cpus, coco))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
