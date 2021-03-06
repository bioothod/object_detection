import os
import json
import logging
import time

import numpy as np

from collections import defaultdict, OrderedDict

import error

logger = logging.getLogger('detection')

class generator(object):
    def __init__(self, ann_file, split_to=1, use_chunk=0, max_min_ratio=50., seed=None):
        np.random.seed(seed)

        self.max_min_ratio = max_min_ratio

        with open(ann_file, 'r') as fin:
            js = json.load(fin)

        # list of structures
        self.annotations = js['annotations']
        self.image_annotations = js['image_annotations']

        self.categories = js['categories']
        self.category_ids = [cat['id'] for cat in self.categories]
        self.image_categories = js['image_categories']
        self.image_category_ids = [cat['id'] for cat in self.image_categories]

        # image_id to structure
        self.images = {img['id']:img for img in js['images']}

        self.cat2ann, self.img2ann = self.parse_annotations(self.annotations, split_to, use_chunk)
        self.whole_cat2ann, self.whole_img2ann = self.parse_annotations(self.image_annotations, split_to, use_chunk)

    def parse_annotations(self, annotations, split_to, use_chunk):
        cat2ann = defaultdict(list)
        img2ann = defaultdict(list)

        annotations = np.array_split(annotations, split_to)[use_chunk]

        for ann in annotations:
            image_id = ann['image_id']
            cat_id = ann['category_id']

            cat2ann[cat_id].append(ann)
            img2ann[image_id].append(ann)

        return cat2ann, img2ann

    def num_images(self):
        return len(self.images)

    def num_classes(self):
        return len(self.categories)
    def num_image_classes(self):
        return len(self.image_categories)

    def get_images_from_categories(self, category_ids, cat2ann, max_num):
        cat_counts = []
        cat_total = 0
        for cat_id in category_ids:
            anns = cat2ann[cat_id]
            ann_num = len(anns)

            cat_counts.append(ann_num)
            cat_total += ann_num

        cat_counts = np.array(cat_counts, dtype=np.float32)
        cat_max = cat_counts.max()
        cat_min = cat_counts.min()

        M = min(self.max_min_ratio, cat_max / cat_min)
        K = (1 - cat_min / cat_max) * M / (M - 1)

        new_counts = []
        for count in cat_counts:
            new_count = cat_max - (cat_max - count) / K
            new_counts.append(new_count)

        new_counts = np.array(new_counts)
        cat_probs = new_counts / new_counts.sum()

        cat_ids = np.random.choice(category_ids, max_num, replace=True, p=cat_probs)
        cat_nums = defaultdict(int)
        for cat_id in cat_ids:
            cat_nums[cat_id] += 1

        ret_images = set()
        for cat_id, cat_num in cat_nums.items():
            anns = cat2ann[cat_id]
            ann_num = len(anns)

            num = min(cat_num, ann_num)

            anns = np.random.choice(anns, num, replace=False)
            for ann in anns:
                image_id = ann['image_id']
                ret_images.add(image_id)

        return ret_images

    def get(self, num, want_full=False):
        pairs = []

        if want_full:
            num = len(self.images)
            images = self.images.keys()
        else:
            cat_images = self.get_images_from_categories(self.category_ids, self.cat2ann, num)
            image_cat_images = self.get_images_from_categories(self.image_category_ids, self.whole_cat2ann, num)
            images = cat_images | image_cat_images

        for image_id in images:
            img = self.images[image_id]
            filename = img['file_name']

            anns = self.img2ann.get(image_id, [])
            image_anns = self.whole_img2ann.get(image_id, [])

            pairs.append((filename, image_id, anns, image_anns))

        #np.random.seed(os.urandom(128))
        np.random.shuffle(pairs)

        ret_filenames = []
        ret_image_ids = []
        ret_anns = []
        ret_image_anns = []

        for fn, image_id, anns, image_anns in pairs:
            ret_filenames.append(fn)
            ret_image_ids.append(image_id)
            ret_anns.append(anns)
            ret_image_anns.append(image_anns)

        return ret_filenames, ret_image_ids, ret_anns, ret_image_anns
