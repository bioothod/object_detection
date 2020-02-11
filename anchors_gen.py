import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

num_scales = 3
num_anchors = 1

def create_anchors():
    anchors_dict = {
        '0': [(10, 13), (30, 61), (116, 90)],
    }

    anchors = anchors_dict["0"]
    # _,_,W,H format
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

    anchors_reshaped = tf.reshape(anchor_boxes, [num_scales, num_anchors, 2])
    anchors_abs_coords = []

    output_xy_grids = []
    output_ratios = []

    for base_scale, output_size in zip(range(num_scales), output_sizes):
        output_size_float = tf.cast(output_size, dtype)
        ratio = float(image_size) / output_size_float

        anchors_wh_one = anchors_reshaped[num_scales - base_scale - 1, ...]
        anchors_wh = tf.expand_dims(anchors_wh_one, 0)
        anchors_wh = tf.tile(anchors_wh, [output_size * output_size, 1, 1])
        anchors_wh = tf.reshape(anchors_wh, [output_size, output_size, num_anchors, 2]) # [13, 13, num_anchors, 2]

        anchors_xy = create_xy_grid(1, output_size, num_anchors)
        anchors_xy = tf.cast(anchors_xy, dtype)
        anchors_xy = tf.squeeze(anchors_xy, 0) # [13, 13, num_anchors, 2]
        anchors_xy_flat = tf.reshape(anchors_xy, [-1, 2])
        output_xy_grids.append(anchors_xy_flat)

        anchors_xy_centers = anchors_xy + 0.5 # centers
        #anchors_xy_centers *= ratio
        anchors_xy *= ratio
        ratios = tf.tile([ratio], [tf.shape(anchors_xy_flat)[0]])
        output_ratios.append(ratios)

        #anchors_for_layer = tf.concat([anchors_xy_centers, anchors_wh], axis=-1)
        anchors_for_layer = tf.concat([anchors_xy, anchors_wh], axis=-1)

        anchors_flat = tf.reshape(anchors_for_layer, [-1, 4])

        logger.info('base_scale: {}: output_size: {}, anchors_for_layer: {}, anchors_flat: {}'.format(base_scale, output_size, anchors_for_layer.shape, anchors_flat.shape))
        anchors_abs_coords.append(anchors_flat)

    anchors_all = tf.concat(anchors_abs_coords, axis=0)
    output_xy_grids = tf.concat(output_xy_grids, axis=0)
    output_ratios = tf.concat(output_ratios, axis=0)

    return anchors_all, output_xy_grids, output_ratios

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
    bboxes = polygon2bbox(poly)

    ious = box_iou(bboxes, anchors_all)
    #logger.info('ious: {}, bboxes: {}, anchors_all: {}'.format(
    #    ious.shape, bboxes.shape, anchors_all.shape))

    best_anchors_index = tf.argmax(ious, 1)

    # grid's shape [N, 2]
    best_anchors = tf.gather(anchors_all[..., :2], best_anchors_index)

    # [N, 2] -> [N, 1, 2]
    best_anchors = tf.expand_dims(best_anchors, 1)

    # [N, 1, 2] -> [N, 4, 2]
    best_anchors = tf.tile(best_anchors, [1, 4, 1])

    # [N, 4, 2] - [N, 4, 2]
    poly_for_loss = poly - best_anchors

    # [N, 4, 2] -> [N, 8]
    poly_for_loss = tf.reshape(poly_for_loss, [-1, 8])
    return poly_for_loss, best_anchors_index

def generate_true_values_for_anchors(word_poly, anchors_all, text_labels, text_lenghts, max_word_len):
    # input polygon shape [N, 4, 2]
    # output polygon shape [N, 4, 2]
    word_poly_for_loss, word_index = find_bbox_anchor_for_poly(word_poly, anchors_all)

    word_objs = tf.ones((tf.shape(word_poly_for_loss)[0], 1), dtype=word_poly.dtype)

    word_idx = tf.expand_dims(word_index, 1)

    # word_obj, word_poly, word, length
    output_dims = 1 + 2*4 + max_word_len + 1

    num_true_anchors = anchors_all.shape[0]
    output = tf.zeros((num_true_anchors, output_dims), dtype=word_poly.dtype)

    text_labels = tf.cast(text_labels, word_objs.dtype)
    text_lenghts = tf.cast(text_lenghts, word_objs.dtype)

    word_values_for_loss = tf.concat([word_objs, word_poly_for_loss, text_labels, text_lenghts], axis=1)

    output = tf.tensor_scatter_nd_update(output, word_idx, word_values_for_loss)

    return output

def unpack_true_values(true_values, all_anchors, current_image_shape, image_size, max_word_len):
    true_word_obj = true_values[..., 0]
    true_word_poly = true_values[..., 1 : 9]
    true_words = true_values[..., 9 : 9 + max_word_len]
    true_lenghts = true_values[..., 9 + max_word_len]

    word_index = tf.where(true_word_obj != 0)

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
