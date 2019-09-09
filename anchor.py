import logging
import math

import numpy as np

logger = logging.getLogger('detection')
logger.setLevel(logging.INFO)

def calc_iou(box, boxes, area):
    bx0 = boxes[:, 0] - boxes[:, 3]/2
    bx1 = boxes[:, 0] + boxes[:, 3]/2
    by0 = boxes[:, 1] - boxes[:, 2]/2
    by1 = boxes[:, 1] + boxes[:, 2]/2

    x0 = box[0] - box[3]/2
    x1 = box[0] + box[3]/2
    y0 = box[1] - box[2]/2
    y1 = box[1] + box[2]/2
    box_area = box[2] * box[3]

    xx0 = np.maximum(x0, bx0)
    yy0 = np.maximum(y0, by0)
    xx1 = np.minimum(x1, bx1)
    yy1 = np.minimum(y1, by1)

    w = np.maximum(0, xx1 - xx0 + 1)
    h = np.maximum(0, yy1 - yy0 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

def create_anchors_for_layer(image_size, layer_size, layer_scale, cell_scales, shifts):
    anchor_boxes, anchor_areas = [], []

    cell_size = image_size * layer_scale

    for x in range(layer_size):
        for y in range(layer_size):
            orig_xc = (x + 0.5) * cell_size
            orig_yc = (y + 0.5) * cell_size

            for scale_h, scale_w in cell_scales:
                w = cell_size * scale_w
                h = cell_size * scale_h

                for dx, dy in shifts:
                    xc = (1 + dx) * orig_xc
                    yc = (1 + dy) * orig_yc

                    x0 = xc - w/2
                    x1 = xc + w/2
                    y0 = yc - h/2
                    y1 = yc + h/2

                    if x0 < 0:
                        x0 = 0
                    if y0 < 0:
                        y0 = 0
                    if x1 > image_size:
                        x1 = image_size
                    if y1 > image_size:
                        y1 = image_size

                    xc = (x0 + x1) / 2
                    yx = (y0 + y1) / 2
                    w = x1 - x0
                    h = y1 - y0

                    bbox = [xc, yc, h, w]
                    area = h * w

                    anchor_boxes.append(bbox)
                    anchor_areas.append(area)

    return anchor_boxes, anchor_areas
