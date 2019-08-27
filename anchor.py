import logging
import math

import numpy as np

logger = logging.getLogger('objdet')
logger.setLevel(logging.INFO)

def calc_iou(box, box_area, boxes, area):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

class Anchor:
    def __init__(self, c0, c1, image_size, layer_size):
        self.c0 = c0
        self.c1 = c1
        self.image_size = image_size
        self.layer_size = layer_size

        self.bbox = self.convert_to_bbox()
        x0, y0, x1, y1 = self.bbox
        self.bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)

    def convert_to_bbox(self):
        scale = self.image_size / self.layer_size

        x0, y0 = self.c0
        x1, y1 = self.c1

        x0 *= scale
        y0 *= scale
        x1 *= scale
        y1 *= scale

        return [x0, y0, x1, y1]

def create_anchors_for_layer(image_size, layer_size, cells_to_side):
    anchor_boxes, anchor_areas = [], []

    for x in range(layer_size):
        for y in range(layer_size):
            for shift in cells_to_side:
                x0 = x - shift
                x1 = x + shift + 1

                y0 = y - shift
                y1 = y + shift + 1

                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0

                if x1 > layer_size:
                    x1 = layer_size
                if y1 > layer_size:
                    y1 = layer_size

                a = Anchor((x0, y0), (x1, y1), image_size, layer_size)

                anchor_boxes.append(a.bbox)
                anchor_areas.append(a.bbox_area)

    return anchor_boxes, anchor_areas

def create_anchors(image_size, feature_shapes):
    anchors = []
    cells_to_side = [0., 0.5, math.sqrt(2), 2.5]
    #cells_to_side = [0.]

    anchor_layers = []
    num_anchors = 0

    anchor_boxes = []
    anchor_areas = []

    for shape in feature_shapes:
        layer_size = shape[1]

        anchor_boxes_for_layer, anchor_areas_for_layer = create_anchors_for_layer(image_size, layer_size, cells_to_side)
        anchor_boxes += anchor_boxes_for_layer
        anchor_areas += anchor_areas_for_layer

        anchor_layers.append(len(anchor_boxes_for_layer))
        num_anchors += len(anchor_boxes_for_layer)

    np_anchors_boxes = np.array(anchor_boxes)
    np_anchors_areas = np.array(anchor_areas)

    return np_anchors_boxes, np_anchors_areas
