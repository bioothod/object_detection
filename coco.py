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

class ProcessingError(Exception):
    def __init__(self, message):
        self.message = message

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

    def num_images(self):
        return len(self.img2anns)

    def cat_names(self):
        d = {}
        for category_id, cat in self.cats.items():
            d[category_id] = cat['name']
        return d

    def gen_anns_for_id(self, image_id, annotations):
        anns = []
        for ann in annotations:
            iscrowd = ann.get('iscrowd', 0)
            if iscrowd != 0:
                continue

            bbox = ann['bbox']
            category_id = ann['category_id']
            segm = ann['segmentation']

            anns.append((bbox, category_id))

        img = self.imgs[image_id]
        filename = os.path.join(self.data_dir, img['file_name'])

        return (filename, image_id, anns)

    def get_images(self):
        tuples = []
        for image_id, annotations in self.img2anns.items():
            t = self.gen_anns_for_id(image_id, annotations)
            tuples.append(t)

        random.shuffle(tuples)

        if False:
            image_id = 283290
            annotations = self.img2anns[image_id]
            t = self.gen_anns_for_id(image_id, annotations)

            self.logger.info('image_id: {}, anns: {}'.format(image_id, t))

            tuples = [t] + tuples

        return tuples

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(image_size, bbox_params):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),

        A.OneOf([
            A.Compose([
                A.Resize(height=int(image_size*1.4), width=int(image_size*1.4), interpolation=cv2.INTER_LANCZOS4, always_apply=True),
                A.RandomCrop(height=image_size, width=image_size, always_apply=True),

                A.IAAAdditiveGaussianNoise(p=0.1),

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
                A.Resize(height=int(image_size*1.2), width=int(image_size*1.2), interpolation=cv2.INTER_LANCZOS4, always_apply=True),
                A.RandomCrop(height=image_size, width=image_size, always_apply=True),
                A.OneOf(
                    [
                        A.IAASharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.5,
                ),
            ]),

            A.Compose([
                A.Resize(height=int(image_size*1.1), width=int(image_size*1.1), interpolation=cv2.INTER_LANCZOS4, always_apply=True),
                A.RandomCrop(height=image_size, width=image_size, always_apply=True),
            ]),
        ], p=1.),
    ]

    return A.Compose(train_transform, bbox_params)


def get_validation_augmentation(image_size, bbox_params):
    test_transform = [
        A.PadIfNeeded(image_size, image_size),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
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
    def __init__(self, ann_path, data_dir, logger, np_anchor_boxes=None, np_anchor_areas=None):
        self.logger = logger

        self.np_anchor_boxes = np_anchor_boxes
        self.np_anchor_areas = np_anchor_areas

        self.image_size = None

        self.train_augmentation = None
        self.eval_augmentation = None
        self.preprocessing = None

        self.coco = COCO(ann_path, data_dir, logger)
        self.image_tuples = self.coco.get_images()

        self.good_bboxes = 0
        self.failed_bboxes = 0

        self.cats = {}
        self.cats[-1] = len(self.cats)
        self.background_id = self.cats[-1]

        for cat_id in self.coco.cats.keys():
            self.cats[cat_id] = len(self.cats)

    def num_classes(self):
        return len(self.cats)

    def set_augmentation(self, image_size=None, train_augmentation=None, eval_augmentation=None, preprocessing=None):
        if not self.image_size:
            self.image_size = image_size

        if not self.train_augmentation:
            self.train_augmentation = train_augmentation

        if not self.eval_augmentation:
            self.eval_augmentation = eval_augmentation

        if not self.preprocessing:
            self.preprocessing = preprocessing

    def set_anchors(self, np_anchor_boxes=None, np_anchor_areas=None):
        if self.np_anchor_boxes is None:
            self.np_anchor_boxes = np_anchor_boxes

        if self.np_anchor_areas is None:
            self.np_anchor_areas = np_anchor_areas

    def cat_names(self):
        return self.coco.cat_names()

    def __len__(self):
        return len(self.image_tuples)

    def process_image(self, i, augmentation, return_orig_format=False):
        start_time = time.time()

        filename, image_id, anns = self.image_tuples[i]

        orig_image = cv2.imread(filename)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image = orig_image.astype(np.uint8)

        orig_bboxes = []
        orig_cat_ids = []
        for bb, cat_id in anns:
            if bb[2] <= 4 or bb[3] <= 4:
                continue

            orig_bboxes.append(bb)
            orig_cat_ids.append(cat_id)

        if len(orig_bboxes) == 0:
            raise ProcessingError('no orig bboxes')

        start_aug_time = time.time()
        annotations = {
            'image': orig_image,
            'bboxes': orig_bboxes,
            'category_id': orig_cat_ids,
        }

        annotations = augmentation(**annotations)
        annotations = self.preprocessing(**annotations)

        image = annotations['image']
        bboxes = annotations['bboxes']
        cat_ids = annotations['category_id']

        if return_orig_format:
            if len(bboxes) == 0:
                raise ProcessingError('there are no bboxes after preprocessing')

            bboxes = np.array(bboxes)
            x0 = np.array(bboxes[:, 0])
            y0 = np.array(bboxes[:, 1])
            w = np.array(bboxes[:, 2])
            h = np.array(bboxes[:, 3])

            cx = x0 + w/2
            cy = y0 + h/2
            true_bboxes = np.stack([cx, cy, h, w], axis=1)

            #self.logger.info('x0: {}, y0: {}, cx: {}, cy: {}, true_bboxes: {}'.format(x0, y0, cx, cy, true_bboxes))
            #exit(-1)

            true_labels = np.array([self.cats[cat_id] for cat_id in cat_ids])

            #self.logger.info('{}: classes: {}\n{}'.format(filename, true_labels, true_bboxes))
            return filename, image_id, image, true_bboxes, true_labels

        max_ious = np.zeros((self.np_anchor_boxes.shape[0]), dtype=np.float32)
        max_per_bbox_ious = np.zeros((self.np_anchor_boxes.shape[0]), dtype=np.float32)

        true_bboxes = self.np_anchor_boxes.copy()
        true_labels = np.zeros((self.np_anchor_boxes.shape[0]))

        def update_true_arrays(filename, image_id, image, box, iou, cat_id, max_iou_threshold, last):
            converted_cat_id = self.cats[cat_id]

            # only select anchor index to update if appropriate anchors have large IoU and IoU for this anchor is larger than that for previous true boxes,
            # and if previous (smaller intersections) were not in fact maximum intersection
            idx = iou > max_iou_threshold
            #update_idx = idx & (iou > max_ious) & (max_per_bbox_ious == 0)
            update_idx = idx & (iou > max_ious)
            if last and False:
                if update_idx.sum() == 0:
                    update_idx = idx & (iou > max_ious)

            assert update_idx.shape == max_ious.shape

            binary_update_idx = update_idx.astype(int)
            masked_iou = iou * binary_update_idx
            max_update = masked_iou.argmax()
            max_per_bbox_ious[max_update] = iou[max_update]

            true_bboxes[update_idx] = box
            true_labels[update_idx] = converted_cat_id
            num_p = update_idx.sum()

            self.logger.debug('{}: image_id: {}, image: {}, bbox: {}, threshold: {}, positive: {}, update: {}, max_iou: {}, max_saved_iou: {}'.format(
                filename, image_id, image.shape, box, max_iou_threshold,
                idx.sum(), update_idx.sum(),
                np.max(iou), np.max(max_ious)))

            max_ious[update_idx] = iou[update_idx]
            return num_p

        good_bboxes = []
        for bb, cat_id in zip(bboxes, cat_ids):
            if bb[2] <= 3 or bb[3] <= 3:
                continue

            good_bboxes.append(bb)
            x0, y0, w, h = bb

            cx = x0 + w/2
            cy = y0 + h/2

            box = np.array([cx, cy, h, w])

            iou = anchor.calc_iou(box, self.np_anchor_boxes, self.np_anchor_areas)

            assert iou.shape == max_ious.shape

            accepted_ious = [0.5]
            for accepted_iou in accepted_ious:
                num_p = update_true_arrays(filename, image_id, image, box, iou, cat_id, accepted_iou, accepted_iou == accepted_ious[-1])
                if num_p != 0:
                    break

        if true_labels.sum() != 0:
            self.good_bboxes += 1
            return filename, image_id, image, true_bboxes, true_labels

        if len(good_bboxes) > 0:
            self.failed_bboxes += 1

            if self.failed_bboxes % 1000 == 0:
                total_bboxes = self.good_bboxes + self.failed_bboxes
                self.logger.info('good_bboxes: {}/{:.4f}, failed_bboxes: {}/{:.4f}, total_bboxes: {}'.format(
                    self.good_bboxes, self.good_bboxes/total_bboxes, self.failed_bboxes, self.failed_bboxes/total_bboxes, total_bboxes))

            areas = [bb[2] * bb[3] for bb in good_bboxes]

            self.logger.debug('{}: image_id: {}, image: {}, bboxes: {}, labels: {}, aug bboxes: {} -> {}/{}, num_positive: {}, num_negatives: {}, time: {:.1f}/{:.1f} ms, bboxes: {}, areas: {}'.format(
                    filename, image_id, image.shape,
                    true_bboxes.shape, true_labels.shape,
                    len(anns), len(bboxes), len(good_bboxes),
                    np.where(true_labels > 0)[0].shape[0], np.where(true_labels == 0)[0].shape[0],
                    (time.time() - start_aug_time) * 1000.,
                    (time.time() - start_time) * 1000.,
                    good_bboxes, areas))

        raise ProcessingError('there are no true labels')

    def __getitem__(self, i):
        if self.np_anchor_areas is None or self.np_anchor_boxes is None:
            raise ValueError('COCO iterable is not initialized: np_anchor_areas: {}, np_anchor_boxes: {}'.format(self.np_anchor_areas, self.np_anchor_boxes))

        augment = self.train_augmentation
        while True:
            try:
                return self.process_image(i, augment)
            except ProcessingError as e:
                if e.message == 'no orig bboxes':
                    i = np.random.randint(len(self.image_tuples))
                else:
                    if augment == self.train_augmentation:
                        augment = self.eval_augmentation
                    else:
                        augment = self.train_augmentation
                        i = np.random.randint(len(self.image_tuples))


def create_coco_iterable(ann_path, data_dir, logger):
    return COCO_Iterable(ann_path, data_dir, logger)

def complete_initialization(coco_base, image_size, np_anchor_boxes, np_anchor_areas, is_training):
    bbox_params = A.BboxParams(
            format='coco',
            min_area=0,
            min_visibility=0.3,
            label_fields=['category_id'])

    if is_training:
	    train_augmentation = get_training_augmentation(image_size, bbox_params)
    else:
	    train_augmentation = get_validation_augmentation(image_size, bbox_params)
    eval_augmentation = get_validation_augmentation(image_size, bbox_params)

    preprocessing = get_preprocessing(preprocess_input, bbox_params)

    coco_base.set_augmentation(image_size, train_augmentation, eval_augmentation, preprocessing)
    coco_base.set_anchors(np_anchor_boxes, np_anchor_areas)
