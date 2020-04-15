import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

num_scales = 3
num_anchors_per_scale = 1

def create_anchors():
    # WH format
    anchors_dict = {
        '0': [(15, 9), (55, 35), (156, 90)],
    }

    anchors = anchors_dict["0"]
    # W,H format
    anchors = np.array(anchors, dtype=np.float32)

    return anchors

def create_xy_grid(batch_size, output_size, anchors_per_scale):
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchors_per_scale, 1])

    return xy_grid

def generate_anchors(image_size, output_sizes, dtype):
    anchor_boxes = create_anchors()
    anchor_boxes = tf.cast(anchor_boxes, dtype)

    anchors_reshaped = tf.reshape(anchor_boxes, [num_scales, num_anchors_per_scale, 2])
    polys = []

    for base_scale, output_size in zip(range(num_scales), output_sizes):
        output_size_float = tf.cast(output_size, dtype)
        ratio = float(image_size) / output_size_float

        anchors_wh_one = anchors_reshaped[num_scales - base_scale - 1, ...]
        anchors_wh = tf.expand_dims(anchors_wh_one, 0) # added batch dimension
        anchors_wh = tf.tile(anchors_wh, [output_size * output_size, 1, 1])
        anchors_wh = tf.reshape(anchors_wh, [output_size, output_size, num_anchors_per_scale, 2]) # [13, 13, num_anchors_per_scale, 2]

        anchors_xy = create_xy_grid(1, output_size, num_anchors_per_scale)
        anchors_xy = tf.cast(anchors_xy, dtype)
        anchors_xy = tf.squeeze(anchors_xy, 0) # [13, 13, num_anchors_per_scale, 2]

        # convert XY from scale grid to absolute grid
        anchors_xy *= ratio

        x0 = anchors_xy[..., 0]
        y0 = anchors_xy[..., 1]
        w = anchors_wh[..., 0]
        h = anchors_wh[..., 1]

        x1 = x0 + w
        y1 = y0
        x2 = x0 + w
        y2 = y0 + w
        x3 = x0
        y3 = y0 + h

        p0 = tf.stack([x0, y0], -1)
        p1 = tf.stack([x1, y1], -1)
        p2 = tf.stack([x2, y2], -1)
        p3 = tf.stack([x3, y3], -1)

        poly = tf.stack([p0, p1, p2, p3], 3)
        poly_flat = tf.reshape(poly, [-1, 4, 2])
        polys.append(poly_flat)

        logger.info('base_scale: {}: output_size: {}, poly: {}'.format(base_scale, output_size, poly.shape))

    anchors_all = tf.concat(polys, axis=0)
    return anchors_all

def box_iou(pred_boxes, valid_true_boxes):
    '''
    param:
        pred_boxes: [N, 4], (xmin, ymin, xmax, ymax)
        valid_true_boxes: [V, 4], (xmin, ymin, w, h)
    '''

    # [N, 2]
    pred_box_min = pred_boxes[..., 0:2]
    pred_box_max = pred_boxes[..., 2:4]

    # [N, 1, 2]
    pred_box_min = tf.expand_dims(pred_box_min, -2)
    pred_box_max = tf.expand_dims(pred_box_max, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[..., 0:2]
    true_box_wh = valid_true_boxes[..., 2:4]

    # [N, 1, 2] & [V, 2] -> [N, V, 2]
    intersect_mins = tf.maximum(pred_box_min, true_box_xy)
    intersect_maxs = tf.minimum(pred_box_max, true_box_xy + true_box_wh)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0)

    # [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # [N, 2]
    pred_box_diff = pred_box_max - pred_box_min

    # [N]
    pred_box_area = pred_box_diff[..., 0] * pred_box_diff[..., 1]

    # [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [N, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou

def polygon2bbox(poly, want_yx=False, want_wh=False):
    # polygon shape [N, 4, 2]

    x = poly[..., 0]
    y = poly[..., 1]

    xmin = tf.math.reduce_min(x, axis=1, keepdims=True)
    ymin = tf.math.reduce_min(y, axis=1, keepdims=True)
    xmax = tf.math.reduce_max(x, axis=1, keepdims=True)
    ymax = tf.math.reduce_max(y, axis=1, keepdims=True)

    if want_yx:
        bbox = tf.concat([ymin, xmin, ymax, xmax], 1)
    elif want_wh:
        bbox = tf.concat([xmin, ymin, xmax - xmin, ymax - ymin], 1)
    else:
        bbox = tf.concat([xmin, ymin, xmax, ymax], 1)

    return bbox

def find_bbox_anchor_for_poly(poly, anchors_all):
    # [N, 4, 2]
    bboxes_min_max = polygon2bbox(poly)
    anchor_bboxes_wh = polygon2bbox(anchors_all, want_wh=True)

    ious = box_iou(bboxes_min_max, anchor_bboxes_wh)
    #logger.info('ious: {}, bboxes: {}, anchors_all: {}'.format(
    #    ious.shape, bboxes.shape, anchors_all.shape))

    best_anchors_index = tf.argmax(ious, 1)

    # anchors shape [N, 4, 2]
    best_anchors = tf.gather(anchors_all, best_anchors_index)

    # [N, 4, 2] - [N, 4, 2]
    poly_for_loss = poly - best_anchors

    # [N, 4, 2] -> [N, 8]
    poly_for_loss = tf.reshape(poly_for_loss, [-1, 8])
    return poly_for_loss, best_anchors_index

def generate_true_values_for_anchors(word_poly, anchors_all, text_labels_list, text_lengths, max_word_len):
    # input polygon shape [N, 4, 2]
    # output polygon shape [N, 4, 2]
    word_poly_for_loss, word_index = find_bbox_anchor_for_poly(word_poly, anchors_all)

    word_objs = tf.ones((tf.shape(word_poly_for_loss)[0], 1), dtype=word_poly.dtype)

    word_idx = tf.expand_dims(word_index, 1)

    # word_obj, word_poly, word, length
    output_dims = 1 + 2*4 + max_word_len*len(text_labels_list) + 1

    num_true_anchors = anchors_all.shape[0]
    output = tf.zeros((num_true_anchors, output_dims), dtype=word_poly.dtype)

    word_values_for_loss = tf.concat([word_objs, word_poly_for_loss] + text_labels_list + [text_lengths], axis=1)

    output = tf.tensor_scatter_nd_update(output, word_idx, word_values_for_loss)

    return output

def unpack_true_values(true_values, all_anchors, current_image_shape, image_size, max_word_len):
    true_word_obj = true_values[..., 0]
    true_word_poly = true_values[..., 1 : 9]
    true_words = true_values[..., 9 : 9 + max_word_len]
    true_lenghts = true_values[..., 9 + max_word_len]

    word_index = tf.where(true_word_obj != 0)

    true_words = tf.gather(true_words, word_index)
    true_lenghts = tf.gather(true_lenghts, word_index)

    word_poly = tf.gather(true_word_poly, word_index)
    word_poly = tf.reshape(word_poly, [-1, 4, 2])

    best_anchors = tf.gather(all_anchors[..., :2], word_index)
    best_anchors = tf.tile(best_anchors, [1, 4, 1])
    word_poly = word_poly + best_anchors

    current_image_shape = tf.cast(current_image_shape, word_poly.dtype)

    imh, imw = current_image_shape[:2]
    max_side = tf.maximum(imh, imw)
    pad_y = (max_side - imh) / 2
    pad_x = (max_side - imw) / 2
    square_scale = max_side / image_size

    word_poly *= square_scale

    diff = [pad_x, pad_y]
    word_poly -= diff

    return true_word_obj, word_poly, true_words, true_lenghts

def create_lookup_table(dictionary):
    pad_value = 0

    dict_split = tf.strings.unicode_split(dictionary, 'UTF-8')
    dictionary_size = dict_split.shape[0] + 1
    kv_init = tf.lookup.KeyValueTensorInitializer(keys=dict_split, values=tf.range(1, dictionary_size, 1), key_dtype=tf.string, value_dtype=tf.int32)
    dict_table = tf.lookup.StaticHashTable(kv_init, pad_value)

    return dictionary_size, dict_table, pad_value
