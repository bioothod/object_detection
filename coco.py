import json
import logging
import os
import random

from collections import defaultdict

import numpy as np

logger = logging.getLogger('objdet')

class COCO:
    def __init__(self, ann_path, data_dir):
        self.data_dir = data_dir

        self.load_instance_json(ann_path)

    def load_instance_json(self, ann_path):
        logging.info('loading instance json {}'.format(ann_path))
        self.dataset = json.load(open(ann_path, 'r'))

        logging.info('generating index from {}'.format(ann_path))
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.img2anns = imgToAnns
        self.cat2mgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        if False:
            num = 0
            for image_id, anns in self.img2anns.items():
                for ann in anns:
                    bbox = ann['bbox']
                    category_id = ann['category_id']
                    category_name = self.cats[category_id]['name']

                    img = self.imgs[image_id]
                    filename = os.path.join(self.data_dir, img['file_name'])
                    logger.info('image: {}, filename: {}, bbox: {}, category: {}/{}'.format(image_id, filename, bbox, category_name, category_id))

                num += 1
                if num > 2:
                    break

            logger.info('annotations: {}, images: {}, categories: {}'.format(len(self.anns), len(self.imgs), len(self.cats)))
            category_names = [c['name'] for c in self.cats.values()]

    def num_classes(self):
        return len(self.cats)

    def num_images(self):
        return len(self.img2anns)

    def cat_names(self):
        d = {}
        for category_id, cat in self.cats.items():
            d[category_id] = cat['name']
        return d

    def get_images(self):
        tuples = []
        for image_id, annotations in self.img2anns.items():
            anns = []
            for ann in annotations:
                bbox = ann['bbox']
                category_id = ann['category_id']

                anns.append((bbox, category_id))

            img = self.imgs[image_id]
            filename = os.path.join(self.data_dir, img['file_name'])

            tuples.append((filename, image_id, anns))

        random.shuffle(tuples)

        filenames = []
        ids = []
        annotations = []
        for t in tuples:
            filename, image_id, anns = t

            filenames.append(filename)
            ids.append(image_id)
            annotations.append(np.array(anns))

        return filenames, ids, annotations
