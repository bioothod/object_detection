import cv2
import json
import os
import random
import time

from collections import defaultdict

import numpy as np

import albumentations as A

class ProcessingError(Exception):
    def __init__(self, message):
        self.message = message

class COCO:
    def __init__(self, ann_path, data_dir, logger):
        self.data_dir = data_dir

        self.logger = logger

        self.load_instance_json(ann_path)

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

            anns.append(ann)

        img = self.imgs[image_id]
        filename = os.path.join(self.data_dir, img['file_name'])

        return (filename, image_id, anns)

    def get_images(self):
        tuples = []
        for image_id, annotations in self.img2anns.items():
            t = self.gen_anns_for_id(image_id, annotations)
            tuples.append(t)

        random.shuffle(tuples)
        return tuples

def get_text_bbox_params():
    return A.BboxParams(
            format='coco',
            min_area=0,
            min_visibility=0.9,
            label_fields=['texts'])

def get_text_train_augmentation(image_size):
    bbox_params = get_text_bbox_params()

    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(rotate_limit=5, p=0.8),

        A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightness(limit=0.01, p=1),
                A.Blur(blur_limit=5, p=1),
                #A.MedianBlur(blur_limit=5, p=1),
                A.RandomContrast(limit=0.01, p=1),
        ], p=0),

    ]

    return A.Compose(train_transform, bbox_params)


def get_text_eval_augmentation(image_size):
    bbox_params = get_text_bbox_params()

    eval_transform = [
        A.PadIfNeeded(image_size, image_size),
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
    ]
    return A.Compose(eval_transform, bbox_params)

class COCOText:
    def __init__(self, ann_path, train_data_dir, eval_data_dir, logger):
        self.logger = logger

        self.load_instance_json(ann_path, train_data_dir, eval_data_dir)

    def load_instance_json(self, ann_path, train_data_dir, eval_data_dir):
        self.dataset = json.load(open(ann_path, 'r'))

        def image_dict():
            return {
                    'anns': [],
            }

        train_images = defaultdict(image_dict)
        eval_images = defaultdict(image_dict)

        skipped_images = 0
        train_anns = 0
        eval_anns = 0

        imgs = self.dataset['imgs']
        anns = self.dataset['anns']
        img2anns = self.dataset['imgToAnns']

        #self.logger.info('dataset keys: {}, images: {}, anns: {}'.format(self.dataset.keys(), len(imgs), len(anns)))

        for image_id_str, image_ctl in imgs.items():
            image_id = int(image_id_str)

            is_training = True
            if image_ctl['set'] == 'val':
                is_training = False
                ctl = eval_images[image_id]
            else:
                ctl = train_images[image_id]

            ctl['image_id'] = image_id

            filename = '{:012d}.jpg'.format(image_id)
            train_fn = os.path.join(train_data_dir, filename)
            eval_fn = os.path.join(eval_data_dir, filename)

            if os.path.exists(train_fn):
                ctl['filename'] = train_fn
            elif os.path.exists(eval_fn):
                ctl['filename'] = eval_fn
            else:
                skipped_images += 1
                continue

            ann_ids = img2anns[image_id_str]

            for ann_id in ann_ids:
                ann = anns[str(ann_id)]

                mask = ann['mask']
                text_class = ann['class']
                bbox = ann['bbox']
                image_id = ann['image_id']
                ann_id = ann['id']
                language = ann['language']
                area = ann['area']
                text = ann['utf8_string']
                legibility = ann['legibility']

                if text == '':
                    ann['utf8_string'] = '<SKIP>'

                ctl['anns'].append(ann)

                if is_training:
                    train_anns += 1
                else:
                    eval_anns += 1

            if is_training:
                train_images[image_id] = ctl
            else:
                eval_images[image_id] = ctl

        self.train_images = list(train_images.values())
        self.eval_images = list(eval_images.values())

        self.logger.info('scanned {}: train: images: {}, anns: {}, eval: images: {}, anns: {}, skipped_images: {}'.format(
            ann_path,
            len(self.train_images), train_anns,
            len(self.eval_images), eval_anns,
            skipped_images))

        image_id = 475485
        def print_text(image_id):
            ctl = train_images.get(image_id)
            if ctl is None:
                ctl = eval_images.get(image_id)
            if ctl is None:
                return None

            texts = []
            for ann in ctl['anns']:
                if ann['legibility'] == 'legible':
                    texts.append(ann['utf8_string'])

            return texts

        #self.logger.info('{}: {}'.format(image_id, print_text(image_id)))

    def num_images(self, is_training):
        if is_training:
            return len(self.train_images)
        else:
            return len(self.eval_images)

    def process_image(self, idx, is_training, augmentation):
        start_time = time.time()

        if is_training:
            image_ctl = self.train_images[idx]
        else:
            image_ctl = self.eval_images[idx]

        filename = image_ctl['filename']
        image_id = image_ctl['image_id']
        anns = image_ctl['anns']

        orig_image = None
        orig_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if orig_image is None:
            self.logger.error('filename: {}, image is none'.format(filename))
            exit(-1)

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image = orig_image.astype(np.uint8)

        orig_bboxes = []
        orig_texts = []
        for ann in anns:
            bb = ann['bbox']
            text = ann['utf8_string']

            if bb[0] < 0:
                bb[0] = 0
            if bb[1] < 0:
                bb[1] = 0

            if bb[0] + bb[2] > orig_image.shape[1]:
                bb[2] = orig_image.shape[1] - bb[0] - 1
            if bb[1] + bb[3] > orig_image.shape[0]:
                bb[3] = orig_image.shape[0] - bb[1] - 1

            orig_bboxes.append(bb)
            orig_texts.append(text)


        if len(orig_bboxes) == 0:
            raise ProcessingError('{}: no orig bboxes'.format(filename))

        start_aug_time = time.time()
        annotations = {
            'image': orig_image,
            'bboxes': orig_bboxes,
            'texts': orig_texts,
        }

        if augmentation:
            annotations = augmentation(**annotations)

        image = annotations['image']
        bboxes = annotations['bboxes']
        texts = annotations['texts']

        if len(bboxes) == 0:
            raise ProcessingError('{}: there are no bboxes after augmentation'.format(filename))

        bboxes = np.array(bboxes)
        x0 = np.array(bboxes[:, 0], dtype=np.float32)
        y0 = np.array(bboxes[:, 1], dtype=np.float32)
        w = np.array(bboxes[:, 2], dtype=np.float32)
        h = np.array(bboxes[:, 3], dtype=np.float32)

        cx = x0 + w/2
        cy = y0 + h/2
        bboxes = np.stack([cx, cy, h, w], axis=1)

        return filename, image_id, image, bboxes, texts

def get_training_augmentation(image_size, bbox_params):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        A.OneOf(
            [
                A.RandomScale(scale_limit=(0.1, 0.3), interpolation=cv2.INTER_CUBIC),
                A.RandomScale(scale_limit=(0.1, 0.5), interpolation=cv2.INTER_CUBIC),
                A.RandomScale(scale_limit=(0.9, 1.1), interpolation=cv2.INTER_CUBIC),
                A.RandomCrop(height=image_size, width=image_size),
        ], p=1),

        A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightness(limit=0.01, p=1),
                A.Blur(blur_limit=5, p=1),
                A.MotionBlur(blur_limit=5, p=1),
                A.MedianBlur(blur_limit=5, p=1),
                A.RandomContrast(limit=0.01, p=1),
        ], p=0.2),

        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        A.OneOf(
            [
                A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
                A.RandomCrop(height=image_size, width=image_size),
        ], p=1),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
    ]

    return A.Compose(train_transform, bbox_params)


def get_validation_augmentation(image_size, bbox_params):
    test_transform = [
        A.PadIfNeeded(image_size, image_size),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return A.Compose(test_transform, bbox_params)

class COCO_Iterable:
    def __init__(self, ann_path, data_dir, logger):
        self.logger = logger

        self.image_size = None

        self.train_augmentation = None
        self.eval_augmentation = None

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

    def set_augmentation(self, image_size=None, train_augmentation=None, eval_augmentation=None):
        if not self.image_size:
            self.image_size = image_size

        if not self.train_augmentation:
            self.train_augmentation = train_augmentation

        if not self.eval_augmentation:
            self.eval_augmentation = eval_augmentation

    def cat_names(self):
        return self.coco.cat_names()

    def __len__(self):
        return len(self.image_tuples)

    def process_image(self, i, augmentation):
        start_time = time.time()

        filename, image_id, anns = self.image_tuples[i]

        orig_image = None
        orig_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if orig_image is None:
            self.logger.error('filename: {}, image is none'.format(filename))
            exit(-1)

        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image = orig_image.astype(np.uint8)

        orig_bboxes = []
        orig_cat_ids = []
        orig_keypoints = []
        for ann in anns:
            bb = ann['bbox']
            cat_id = ann['category_id']

            orig_bboxes.append(bb)
            orig_cat_ids.append(cat_id)
            orig_keypoints += kp

            if bb[2] > orig_image.shape[1] or bb[3] > orig_image.shape[0]:
                self.logger.error('{}: incorrect annotation: image_shape: {}, bb: {}'.format(filename, orig_image.shape, bb))

        if len(orig_bboxes) == 0:
            raise ProcessingError('{}: no orig bboxes'.format(filename))

        start_aug_time = time.time()
        annotations = {
            'image': orig_image,
            'bboxes': orig_bboxes,
            'category_id': orig_cat_ids,
        }

        if augmentation:
            annotations = augmentation(**annotations)

        image = annotations['image']
        bboxes = annotations['bboxes']
        cat_ids = annotations['category_id']

        if len(bboxes) == 0:
            raise ProcessingError('{}: there are no bboxes after augmentation'.format(filename))

        bboxes = np.array(bboxes)
        x0 = np.array(bboxes[:, 0], dtype=np.float32)
        y0 = np.array(bboxes[:, 1], dtype=np.float32)
        w = np.array(bboxes[:, 2], dtype=np.float32)
        h = np.array(bboxes[:, 3], dtype=np.float32)

        cx = x0 + w/2
        cy = y0 + h/2
        true_bboxes = np.stack([cx, cy, h, w], axis=1)

        true_labels = np.array([self.cats[cat_id] for cat_id in cat_ids], dtype=np.int32)

        return filename, image_id, image, true_bboxes, true_labels

def create_coco_iterable(ann_path, data_dir, logger):
    return COCO_Iterable(ann_path, data_dir, logger)

def complete_initialization(coco_base, image_size, is_training, train_augmentation=None):
    bbox_params = A.BboxParams(
            format='coco',
            min_area=0,
            min_visibility=0.6,
            label_fields=['category_id'])

    if train_augmentation is None:
        if is_training:
            train_augmentation = get_training_augmentation(image_size, bbox_params)
        else:
            train_augmentation = get_validation_augmentation(image_size, bbox_params)

    eval_augmentation = get_validation_augmentation(image_size, bbox_params)

    coco_base.set_augmentation(image_size, train_augmentation, eval_augmentation)
