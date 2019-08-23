import cv2
import logging
import os
import time

import albumentations as A
import multiprocessing as mp
import numpy as np

import polygon_dataset

logger = logging.getLogger('segmentation')

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(image_size):
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        #A.Resize(height=image_size*2, width=image_size*2, interpolation=cv2.INTER_CUBIC, always_apply=True),

        A.RandomCrop(height=image_size, width=image_size, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(image_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(image_size, image_size),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

class Dataset:
    def __init__(self, data_dir,
                 augmentation=None, 
                 preprocessing=None
                ):
        good_exts = ['.png', '.jpg', '.jpeg']

        self.image_filenames = []
        self.json_filenames = []
        self.cards = []

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        for fn in os.listdir(data_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in good_exts:
                continue
            
            image_filename = os.path.join(data_dir, fn)
            json_filename = os.path.join(data_dir, '{}.json'.format(fn))

            self.image_filenames.append(image_filename)
            self.json_filenames.append(json_filename)
            self.cards.append(None)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, i):
        start_time = time.time()
        img_fn = self.image_filenames[i]
        js_fn = self.json_filenames[i]

        image = cv2.imread(img_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        if self.cards[i] == None:
            card = polygon_dataset.Card(js_fn, img_fn, image.shape[0], image.shape[1])
            self.cards[i] = card
        else:
            card = self.cards[i]

        mask = np.concatenate([card.card_mask, card.text_mask], axis=-1)

        background = 1. - np.where(mask.sum(axis=-1, keepdims=True) > 0, 1, 0)
        mask = np.concatenate([background, mask], axis=-1)

        mask = mask.astype(np.uint8)

        si_shape = image.shape
        si_dtype = image.dtype
        sm_shape = mask.shape
        sm_dtype = mask.dtype

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        logger.debug('{}: {}: img: {}/{} -> {}/{}, mask: {}/{} -> {}/{}, time: {:.1f} ms'.format(
            os.getpid(), img_fn,
            si_shape, si_dtype, image.shape, image.dtype,
            sm_shape, sm_dtype, mask.shape, mask.dtype,
            (time.time() - start_time) * 1000))

        return img_fn, image, mask

def preprocess_input(img, **kwargs):
    #logger.info('preprocess: img: {}/{}, kwargs: {}'.format(img.shape, img.dtype, kwargs))

    img = img.astype(np.float32)

    #vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
    #img -= vgg_means
    #img /= 255.
    img -= 128.
    img /= 128.

    return img

def run_queue(num_processes, data_dir, image_size):
    dataset = Dataset(data_dir, augmentation=get_training_augmentation(image_size), preprocessing=get_preprocessing(preprocess_input))
    return dataset

def empty_callable(x):
    return x
