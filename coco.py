import json
import os
import random

from collections import defaultdict

import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pycocotools import mask as maskUtils

class COCO:
    def __init__(self, ann_path, data_dir, logger):
        self.data_dir = data_dir

        self.logger = logger

        self.load_instance_json(ann_path)

    def ann2rle(self, img, ann):
        if not 'segmentation' in ann:
            return None

        if type(ann['segmentation']) == list:
            segm = ann['segmentation']
            # polygon
            #for seg in segm:
            #    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                #polygons.append(Polygon(poly))
                #color.append(c)

            rles = maskUtils.frPyObjects(segm, img['height'], img['width'])
            rle = maskUtils.merge(rles)
        else:
            # mask
            if type(ann['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects([ann['segmentation']], img['height'], img['width'])
            else:
                rle = [ann['segmentation']]

        return rle

    def ann2mask(self, img, ann):
        rle = self.ann2rle(img, ann)
        if rle is None:
            return None

        return maskUtils.decode(rle)

    def merge_masks(self, img, anns):
        rles = []
        for ann in anns:
            rles.append(self.ann2rle(img, ann))

        maskUtils.merge(rles)

    def load_instance_json(self, ann_path):
        self.dataset = json.load(open(ann_path, 'r'))

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

                    segm_mask = self.ann2mask(img, ann)
                    segm_mask_present = segm_mask is not None

                    self.logger.info('image: {}, {}x{}, filename: {}, bbox: {}, category: {}/{}, segm mask: {}'.format
                            (image_id, img['height'], img['width'], filename, bbox, category_name, category_id, segm_mask_present))

                num += 1
                if num > 2:
                    break

            self.logger.info('annotations: {}, images: {}, categories: {}'.format(len(self.anns), len(self.imgs), len(self.cats)))
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
                segm = ann['segmentation']

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
