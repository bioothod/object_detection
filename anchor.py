import logging
import math

import numpy as np

logger = logging.getLogger('detection')
logger.setLevel(logging.INFO)

def calc_iou(box, box_area, boxes, area):
    xx1 = np.maximum(box[1], boxes[:, 1])
    yy1 = np.maximum(box[0], boxes[:, 0])
    xx2 = np.minimum(box[3], boxes[:, 3])
    yy2 = np.minimum(box[2], boxes[:, 2])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

def create_anchors_for_layer(image_size, layer_size, cells_to_side, shifts):
    anchor_boxes, anchor_areas = [], []

    scale = image_size / layer_size

    for x in range(layer_size):
        for y in range(layer_size):
            for shift_x, shift_y in cells_to_side:
                x0 = x - shift_x
                x1 = x + shift_x + 1

                y0 = y - shift_y
                y1 = y + shift_y + 1

                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0

                if x1 > layer_size:
                    x1 = layer_size
                if y1 > layer_size:
                    y1 = layer_size

                x0 *= scale
                y0 *= scale
                x1 *= scale
                y1 *= scale

                for dx, dy in shifts:
                    x0 += dx*x0
                    x1 += dx*x1
                    y0 += dy*y0
                    y1 += dy*y1

                    if x0 < 0:
                        x0 = 0
                    if y0 < 0:
                        y0 = 0
                    if x1 > image_size:
                        x1 = image_size
                    if y1 > image_size:
                        y1 = image_size

                    bbox = [y0, x0, y1, x1]
                    area = (x1 - x0 + 1) * (y1 - y0 + 1)

                    anchor_boxes.append(bbox)
                    anchor_areas.append(area)

    return anchor_boxes, anchor_areas
