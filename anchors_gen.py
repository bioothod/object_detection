import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

def calc_iou_for_single_box(box, bx0, by0, bx1, by1, areas):
    x0 = box[0] - box[2]/2
    x1 = box[0] + box[2]/2
    y0 = box[1] - box[3]/2
    y1 = box[1] + box[3]/2
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
    bx0 = boxes[:, 0] - boxes[:, 2]/2
    bx1 = boxes[:, 0] + boxes[:, 2]/2
    by0 = boxes[:, 1] - boxes[:, 3]/2
    by1 = boxes[:, 1] + boxes[:, 3]/2

    ious = tf.map_fn(lambda box: calc_iou_for_single_box(box, bx0, by0, bx1, by1, areas), orig_bboxes, parallel_iterations=32)
    return ious

def calc_ious_one_to_one(pred_bboxes, true_bboxes):
    px0 = pred_bboxes[..., 0] - pred_bboxes[..., 2]/2
    px1 = pred_bboxes[..., 0] + pred_bboxes[..., 2]/2
    py0 = pred_bboxes[..., 1] - pred_bboxes[..., 3]/2
    py1 = pred_bboxes[..., 1] + pred_bboxes[..., 3]/2
    pareas = pred_bboxes[..., 2] * pred_bboxes[..., 3]

    tx0 = true_bboxes[..., 0] - true_bboxes[..., 2]/2
    tx1 = true_bboxes[..., 0] + true_bboxes[..., 2]/2
    ty0 = true_bboxes[..., 1] - true_bboxes[..., 3]/2
    ty1 = true_bboxes[..., 1] + true_bboxes[..., 3]/2
    tareas = true_bboxes[..., 2] * true_bboxes[..., 3]

    xx0 = tf.maximum(px0, tx0)
    yy0 = tf.maximum(py0, ty0)
    xx1 = tf.minimum(px1, tx1)
    yy1 = tf.minimum(py1, ty1)

    w = tf.maximum(0., xx1 - xx0 + 1)
    h = tf.maximum(0., yy1 - yy0 + 1)

    inter = w * h
    iou = inter / (pareas + tareas - inter)
    return iou

DOWNSAMPLE_RATIO = 32

def generate_true_labels_for_anchors(orig_bboxes, orig_labels, np_anchor_boxes, np_anchor_areas, image_size, num_classes):
    num_anchors = np_anchor_boxes.shape[0]

    orig_bboxes = tf.convert_to_tensor(orig_bboxes, dtype=tf.float32)
    orig_labels = tf.convert_to_tensor(orig_labels, dtype=tf.int32)

    _, _, w, h = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)
    shifted_orig_bboxes = tf.concat([tf.zeros_like(w), tf.zeros_like(h), w, h], axis=1)
    anchor_bboxes = tf.concat([tf.zeros((num_anchors, 2), tf.float32), np_anchor_boxes], axis=1)

    num_scales = 3

    ious = calc_ious(shifted_orig_bboxes, anchor_bboxes, np_anchor_areas)
    anchor_idx = tf.argmax(ious, axis=1)
    box_idx = tf.math.floormod(anchor_idx, num_scales)
    scale_idx = tf.math.floordiv(anchor_idx, num_scales)

    #tf.print('bboxes:', orig_bboxes, 'ious:', ious, 'anchor_idx:', anchor_idx, 'box_idx:', box_idx, 'scale_idx', scale_idx)

    #anchor_idx = tf.cast(anchor_idx, tf.int64)
    #anchor_idx_range = tf.stack([anchor_idx, tf.range(num_anchors, dtype=tf.int64)], axis=1)


    matched_anchor_boxes = tf.gather(np_anchor_boxes, anchor_idx)


    scaled_size = tf.math.floordiv(image_size, DOWNSAMPLE_RATIO)

    ret = []
    for base_scale in range(num_scales):
        output_size = scaled_size * tf.math.pow(2, base_scale)
        output_size_float = tf.cast(output_size, tf.float32)

        scale_match_idx_orig = tf.where(tf.equal(scale_idx, base_scale))
        scale_match_idx = tf.squeeze(scale_match_idx_orig, 1)
        orig_bboxes_matched_scale = tf.gather(orig_bboxes, scale_match_idx)
        orig_labels_matched_scale = tf.gather(orig_labels, scale_match_idx)

        logger.info('base_scale: {}, orig_bboxes: {}, anchor_idx: {}, scale_idx: {}, scale_match_idx: {}, orig_bboxes_matched_scale: {}'.format(
            base_scale, orig_bboxes.shape, anchor_idx.shape, scale_idx.shape, scale_match_idx.shape, orig_bboxes_matched_scale.shape))

        cx, cy, w, h = tf.split(orig_bboxes_matched_scale, num_or_size_splits=4, axis=1)

        out_cx = cx / float(image_size) * output_size_float
        out_cy = cy / float(image_size) * output_size_float

        anchors_for_scale = tf.gather(matched_anchor_boxes, scale_match_idx)
        anchor_w, anchor_h = tf.split(anchors_for_scale, num_or_size_splits=2, axis=1)

        #tf.print('anchor_idx:', tf.shape(anchor_idx), 'h:', tf.shape(h), 'ah:', tf.shape(anchor_h))
        out_w = tf.math.log(tf.maximum(w, 1) / anchor_w)
        out_h = tf.math.log(tf.maximum(h, 1) / anchor_h)

        bboxes_to_set = tf.concat([out_cx, out_cy, out_w, out_h], axis=1)
        #bboxes_to_set = tf.concat([cx, cy, w, h], axis=1)
        #bboxes_to_set = tf.concat([out_cx, out_cy, w, h], axis=1)
        obj_to_set = tf.ones_like(cx)
        labels_to_set = tf.one_hot(orig_labels_matched_scale, num_classes, dtype=tf.float32)

        values_to_set = tf.concat([bboxes_to_set, obj_to_set, labels_to_set], axis=1)
        logger.info('base_scale: {}, concat: bboxes: {}, obj: {}, labels: {}, values: {}'.format(base_scale, bboxes_to_set, obj_to_set, labels_to_set, values_to_set))

        icx = tf.cast(tf.floor(out_cx), tf.int64)
        icy = tf.cast(tf.floor(out_cy), tf.int64)

        box_match_idx = tf.gather(box_idx, scale_match_idx_orig)

        update_idx = tf.concat([icy, icx, box_match_idx], axis=1)

        #tf.print('update_idx:', update_idx, 'values_to_set:', values_to_set)

        output = tf.zeros((output_size, output_size, num_scales, 4+1+num_classes), dtype=tf.float32)

        logger.info('base_scale: {}, output: {}, update_idx: {}, values_to_set: {}'.format(base_scale, output.shape, update_idx.shape, values_to_set.shape))
        output = tf.tensor_scatter_nd_update(output, update_idx, values_to_set)

        output = tf.reshape(output, [output.shape[0] * output.shape[1], output.shape[2], output.shape[3]])
        ret.append(output)

    ret = tf.concat(ret, axis=0)
    return ret
