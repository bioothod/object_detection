import anchor
import cv2
import json
import os
import random
import time

from collections import defaultdict

import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pycocotools import mask as maskUtils

import albumentations as A

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

        return tuples

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(image_size, bbox_params):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),

        A.OneOf([
            A.Compose([
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

                A.Resize(height=int(image_size*1.6), width=int(image_size*1.6), interpolation=cv2.INTER_CUBIC, always_apply=True),
                A.RandomCrop(height=image_size, width=image_size, always_apply=True),

                A.IAAAdditiveGaussianNoise(p=0.1),
                A.IAAPerspective(p=0.1),

                A.OneOf(
                    [
                        A.CLAHE(p=1),
                        A.RandomBrightness(p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.5,
                ),

                A.OneOf(
                    [
                        A.IAASharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.5,
                ),

                A.OneOf(
                    [
                        A.RandomContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.5,
                ),
                A.Lambda(mask=round_clip_0_1),
            ]),
            A.Compose([
                A.Resize(height=int(image_size*1.2), width=int(image_size*1.2), interpolation=cv2.INTER_CUBIC, always_apply=True),
                A.RandomCrop(height=image_size, width=image_size, always_apply=True),
            ]),
        ], p=1.),
    ]

    return A.Compose(train_transform, bbox_params)


def get_validation_augmentation(image_size, bbox_params):
    test_transform = [
        A.PadIfNeeded(image_size, image_size),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return A.Compose(test_transform, bbox_params)

def preprocess_input(img, **kwargs):
    #logger.info('preprocess: img: {}/{}, kwargs: {}'.format(img.shape, img.dtype, kwargs))

    img = img.astype(np.float32)

    #vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
    #img -= vgg_means
    #img /= 255.
    img -= 128.
    img /= 128.

    return img

def get_preprocessing(preprocessing_fn, bbox_params):
    transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(transform, bbox_params)

class COCO_Iterable:
    def __init__(self, ann_path, data_dir, logger, np_anchor_boxes, np_anchor_areas, augmentation=None, preprocessing=None):
        self.logger = logger

        self.np_anchor_boxes = np_anchor_boxes
        self.np_anchor_areas = np_anchor_areas

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.coco = COCO(ann_path, data_dir, logger)
        self.image_tuples = self.coco.get_images()

        self.cats = {}
        self.cats[-1] = len(self.cats)
        self.background_id = self.cats[-1]

        for cat_id in self.coco.cats.keys():
            self.cats[cat_id] = len(self.cats)

        self.orig_images = {}

    def num_classes(self):
        return self.coco.num_classes()

    def cat_names(self):
        return self.coco.cat_names()

    def __len__(self):
        return len(self.image_tuples)

    def __getitem__(self, i):
        start_time = time.time()

        filename, image_id, anns = self.image_tuples[i]

        image = self.orig_images.get(i, None)
        if image is None:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.uint8)

            self.orig_images[i] = image

        bboxes = []
        cat_ids = []
        for bb, cat_id in anns:
            if bb[2] <= 1 or bb[3] <= 1:
                #self.logger.error('{}: image_id: {}, image: {}: bad bbox: {}, bboxes: {}, categories: {}'.format(filename, image_id, image.shape, bb, bboxes, cat_ids))
                continue

            bboxes.append(bb)
            cat_ids.append(cat_id)

        annotations = {
            'image': image,
            'bboxes': bboxes,
            'category_id': cat_ids,
        }

        #self.logger.info('{}: image_id: {}, image: {}: before processing: bboxes: {}, categories: {}'.format(filename, image_id, image.shape, bboxes, cat_ids))

        if self.augmentation:
            annotations = self.augmentation(**annotations)

        if self.preprocessing:
            annotations = self.preprocessing(**annotations)

        image = annotations['image']
        bboxes = annotations['bboxes']
        cat_ids = annotations['category_id']

        #self.logger.info('{}: image_id: {}, image: {}: after preprocessing: bboxes: {}, categories: {}'.format(filename, image_id, image.shape, bboxes, cat_ids))

        max_ious = np.zeros((self.np_anchor_boxes.shape[0]), dtype=np.float32)

        true_bboxes = self.np_anchor_boxes.copy()
        true_labels = np.zeros((self.np_anchor_boxes.shape[0]))
        true_orig_labels = np.zeros((self.np_anchor_boxes.shape[0]))
        num_positive = 0

        def update_true_arrays(iou, cat_id, max_iou_threshold):
            converted_cat_id = self.cats[cat_id]

            idx = iou > max_iou_threshold
            update_idx = idx & (iou > max_ious)

            true_bboxes[update_idx] = box
            true_labels[update_idx] = converted_cat_id
            true_orig_labels[update_idx] = cat_id
            num_p = update_idx.sum()

            self.logger.debug('{}: image_id: {}, image: {}, bbox: {}, threshold: {}, positive: {}, update: {}, max_iou: {}, max_saved_iou: {}'.format(
                filename, image_id, image.shape, box, max_iou_threshold,
                idx.sum(), update_idx.sum(),
                np.max(iou), np.max(max_ious)))

            max_ious[update_idx] = iou[update_idx]
            return num_p

        for bb, cat_id in zip(bboxes, cat_ids):
            x0, y0, x1, y1 = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

            box = np.array([x0, y0, x1, y1])
            box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            iou = anchor.calc_iou(box, box_area, self.np_anchor_boxes, self.np_anchor_areas)

            assert iou.shape == max_ious.shape

            num_p = update_true_arrays(iou, cat_id, 0.5)
            if num_p == 0:
                max_iou = np.max(iou)
                num_p = update_true_arrays(iou, cat_id, max_iou * 0.8)

            num_positive += num_p

        self.logger.debug('{}: image_id: {}, image: {}, bboxes: {}, labels: {}, aug bboxes: {} -> {}, num_positive: {}, time: {:.1f} ms'.format(
            filename, image_id, image.shape, true_bboxes.shape, true_labels.shape, len(anns), len(bboxes), num_positive, (time.time() - start_time) * 1000.))
        return filename, image_id, image, true_bboxes, true_labels, true_orig_labels

def create_coco_iterable(image_size, ann_path, data_dir, logger, is_training, np_anchor_boxes, np_anchor_areas, min_area=0., min_visibility=0.15):
    bbox_params = A.BboxParams(
            format='coco',
            min_area=min_area,
            min_visibility=min_visibility,
            label_fields=['category_id'])

    if is_training:
        augmentation = get_training_augmentation(image_size, bbox_params)
    else:
        augmentation = get_validation_augmentation(image_size, bbox_params)

    preprocessing = get_preprocessing(preprocess_input, bbox_params)

    return COCO_Iterable(ann_path, data_dir, logger, np_anchor_boxes, np_anchor_areas, augmentation, preprocessing)
