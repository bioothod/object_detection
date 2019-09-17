import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

def calc_iou_for_single_box(box, bx0, by0, bx1, by1, areas):
    x0 = box[0] - box[3]/2
    x1 = box[0] + box[3]/2
    y0 = box[1] - box[2]/2
    y1 = box[1] + box[2]/2
    box_area = box[2] * box[3]

    xx0 = tf.maximum(x0, bx0)
    yy0 = tf.maximum(y0, by0)
    xx1 = tf.minimum(x1, bx1)
    yy1 = tf.minimum(y1, by1)

    w = tf.maximum(0., xx1 - xx0 + 1)
    h = tf.maximum(0., yy1 - yy0 + 1)

    inter = w * h
    iou = inter / (box_area + areas - inter)
    return iou

def calc_ious(orig_bboxes, boxes, areas):
    bx0 = boxes[:, 0] - boxes[:, 3]/2
    bx1 = boxes[:, 0] + boxes[:, 3]/2
    by0 = boxes[:, 1] - boxes[:, 2]/2
    by1 = boxes[:, 1] + boxes[:, 2]/2

    ious = tf.map_fn(lambda box: calc_iou_for_single_box(box, bx0, by0, bx1, by1, areas), orig_bboxes, parallel_iterations=32)
    return ious

def calc_ious_one_to_one(pred_bboxes, true_bboxes):
    px0 = pred_bboxes[0] - pred_bboxes[3]/2
    px1 = pred_bboxes[0] + pred_bboxes[3]/2
    py0 = pred_bboxes[1] - pred_bboxes[2]/2
    py1 = pred_bboxes[1] + pred_bboxes[2]/2
    pareas = pred_bboxes[2] * pred_bboxes[3]

    tx0 = true_bboxes[0] - true_bboxes[3]/2
    tx1 = true_bboxes[0] + true_bboxes[3]/2
    ty0 = true_bboxes[1] - true_bboxes[2]/2
    ty1 = true_bboxes[1] + true_bboxes[2]/2
    tareas = true_bboxes[2] * true_bboxes[3]

    xx0 = tf.maximum(px0, tx0)
    yy0 = tf.maximum(py0, ty0)
    xx1 = tf.minimum(px1, tx1)
    yy1 = tf.minimum(py1, ty1)

    w = tf.maximum(0., xx1 - xx0 + 1)
    h = tf.maximum(0., yy1 - yy0 + 1)

    inter = w * h
    iou = inter / (pareas + tareas - inter)
    return iou


def generate_true_labels_for_anchors(orig_bboxes, orig_labels, np_anchor_boxes, np_anchor_areas):
    num_anchors = np_anchor_boxes.shape[0]

    orig_bboxes = tf.convert_to_tensor(orig_bboxes, dtype=tf.float32)
    orig_labels = tf.convert_to_tensor(orig_labels, dtype=tf.int32)

    true_bboxes = tf.convert_to_tensor(np_anchor_boxes.copy(), dtype=tf.float32)
    true_labels = tf.zeros((num_anchors,), dtype=tf.int32)

    ious = calc_ious(orig_bboxes, np_anchor_boxes, np_anchor_areas)
    orig_dim_idx = tf.argmax(ious, axis=0)
    orig_dim_idx = tf.cast(orig_dim_idx, tf.int64)
    orig_dim_idx_range = tf.stack([orig_dim_idx, tf.range(num_anchors, dtype=tf.int64)], axis=1)

    max_ious = tf.gather_nd(ious, orig_dim_idx_range)

    max_iou_threshold = 0.5
    update_idx = tf.where(max_ious > max_iou_threshold)
    update_idx = tf.squeeze(update_idx, 1)

    orig_idx = tf.gather(orig_dim_idx, update_idx)
    labels_to_set = tf.gather(orig_labels, orig_idx)
    bboxes_to_set = tf.gather(orig_bboxes, orig_idx)
    update_idx = update_idx
    update_idx = tf.expand_dims(update_idx, 1)

    true_labels = tf.tensor_scatter_nd_update(true_labels, update_idx, labels_to_set)
    true_bboxes = tf.tensor_scatter_nd_update(true_bboxes, update_idx, bboxes_to_set)

    return true_bboxes, true_labels
