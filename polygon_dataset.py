import cv2
import json
import logging
import os
import sys

import numpy as np

logger = logging.getLogger('segmentation')

class Card:
    def __init__(self, ann_filename, image_filename, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        self.image_filename = image_filename

        self.card_mask = None
        self.number_text = None
        self.exp_data_text = None
        self.text_mask = None

        self.parse_single(ann_filename)

    def float2abs(self, coord):
        return [coord[0] * self.target_width, coord[1] * self.target_height]

    def find_poly(self, obj):
        poly_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

        poly = [self.float2abs(obj[name]) for name in poly_names]
        return poly

    def parse_single(self, filename):
        js = json.load(open(filename, 'r'))

        card = js['card']
        card_poly = self.find_poly(card)
        card_poly = np.array(card_poly, dtype=np.int32)
        self.card_mask = np.zeros((self.target_height, self.target_width, 1), dtype=np.uint8)
        cv2.fillPoly(self.card_mask, [card_poly], 1)

        self.boundary_mask = np.zeros((self.target_height, self.target_width, 1), dtype=np.uint8)
        cv2.polylines(self.boundary_mask, [card_poly], True, 1)

        self.corners_mask = np.zeros((self.target_height, self.target_width, 1), dtype=np.uint8)
        for i in range(card_poly.shape[0]):
            p = card_poly[i]

            p = (p[0], p[1])
            cv2.circle(self.corners_mask, p, 10, 1, -1)

        number = js['number']
        self.number_text = number['text']
        number_poly = np.array(self.find_poly(number), dtype=np.int32)
        self.text_mask = np.zeros((self.target_height, self.target_width, 1), dtype=np.uint8)
        cv2.fillPoly(self.text_mask, [number_poly], 1)

        exp_date = js['exp_date']
        self.exp_date_text = exp_date['text']
        exp_date_poly = np.array(self.find_poly(exp_date), dtype=np.int32)
        #cv2.fillPoly(self.text_mask, [exp_date_poly], 1)

        #logger.info('{}: number poly: {}, date poly: {}'.format(filename, number_poly, exp_date_poly))

class Polygons:
    def __init__(self, ann_filename, logger, target_height, target_width):
        self.logger = logger

        self.target_height = target_height
        self.target_width = target_width

        good_exts = ['.png', '.jpg', '.jpeg']

        self.cards = []

        for fn in os.listdir(ann_filename):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in good_exts:
                continue
            
            image_filename = os.path.join(ann_filename, fn)
            json_filename = os.path.join(ann_filename, '{}.json'.format(fn))

            try:
                card = Card(json_filename, image_filename, self.target_height, self.target_width)
                self.cards.append(card)
            except Exception as e:
                self.logger.error('could not parse card data: ann_filename: {}, image_filename: {}, exception: {}'.format(json_filename, image_filename, e))

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

    def get_filenames(self):
        return [card.image_filename for card in self.cards]

    def get_masks(self):
        return [card.card_mask for card in self.cards]
