import cv2
import json
import os
import random
import sys

import numpy as np

class Card:
    def __init__(self, ann_path, image_path, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        self.image_path = image_path

        self.card_mask = None
        self.number_text = None
        self.exp_data_text = None
        self.text_mask = None

        self.parse_single(ann_path)

    def float2abs(self, coord):
        return [coord[0] * self.target_width, coord[1] * self.target_height]

    def find_poly(self, obj):
        poly_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

        poly = [self.float2abs(obj[name]) for name in poly_names]
        return poly

    def apply_mask(self, mask, polys, value):
        if mask is None:
            mask = np.zeros((self.target_height, self.target_width, 1), dtype=np.int32)

        polys = np.array(polys, dtype=np.int32)
        cv2.fillPoly(mask, polys, value)
        return mask

    def parse_single(self, path):
        js = json.load(open(path, 'r'))

        card = js['card']
        card_poly = self.find_poly(card)
        self.card_mask = self.apply_mask(None, [card_poly], 1)

        number = js['number']
        self.number_text = number['text']
        number_poly = self.find_poly(number)

        exp_date = js['exp_date']
        self.exp_date_text = exp_date['text']
        exp_date_poly = self.find_poly(exp_date)

        self.card_mask = self.apply_mask(self.card_mask, [number_poly, exp_date_poly], 2)

class Polygons:
    def __init__(self, ann_path, logger, target_height, target_width):
        self.logger = logger

        self.target_height = target_height
        self.target_width = target_width

        good_exts = ['.png', '.jpg', '.jpeg']

        self.cards = []

        for fn in os.listdir(ann_path):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in good_exts:
                continue
            
            image_path = os.path.join(ann_path, fn)
            json_path = os.path.join(ann_path, '{}.json'.format(fn))

            try:
                card = Card(json_path, image_path, self.target_height, self.target_height)
                self.cards.append(card)
            except Exception as e:
                self.logger.error('could not parse card data: ann_path: {}, image_path: {}, exception: {}'.format(json_path, image_path, e))

                exc_type, exc_value, exc_traceback = sys.exc_info()

                logger.error("got error: {}".format(e))

                import traceback

                lines = traceback.format_exc().splitlines()
                for l in lines:
                    logger.info(l)

                traceback.print_exception(exc_type, exc_value, exc_traceback)
                return
                continue
        
    def get_cards(self):
        return self.cards

    def num_images(self):
        return len(self.cards)
