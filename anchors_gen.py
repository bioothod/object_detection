import logging

import numpy as np
import tensorflow as tf

import loss

logger = logging.getLogger('detection')


def create_anchors():
    anchors_dict = {
        '0': [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
        '1': [(17, 18), (28, 24), (36, 34), (42, 44), (56, 51), (72, 66), (90, 95), (92, 154), (139, 281)],
    }

    anchors = anchors_dict["0"]
    # _,_,W,H format
    anchors = np.array(anchors, dtype=np.float32)
    areas = anchors[:, 0] * anchors[:, 1]

    return anchors, areas

def generate_anchors(image_size, output_sizes):
        num_scales = 3
        num_boxes = 3

        np_anchor_boxes, np_anchor_areas = create_anchors()

        anchors_reshaped = tf.reshape(np_anchor_boxes, [num_scales, num_boxes, 2])
        anchors_abs_coords = []

        output_xy_grids = []
        output_ratios = []

        for base_scale, output_size in zip(range(num_scales), output_sizes):
            output_size_float = tf.cast(output_size, tf.float32)
            ratio = float(image_size) / output_size_float

            anchors_wh_one = anchors_reshaped[num_scales - base_scale - 1, ...]
            anchors_wh = tf.expand_dims(anchors_wh_one, 0)
            anchors_wh = tf.tile(anchors_wh, [output_size * output_size, 1, 1])
            anchors_wh = tf.reshape(anchors_wh, [output_size, output_size, num_boxes, 2]) # [13, 13, 3, 2]

            anchors_xy = create_xy_grid(1, output_size, num_boxes)
            anchors_xy = tf.squeeze(anchors_xy, 0) # [13, 13, 3, 2]
            anchors_xy_flat = tf.reshape(anchors_xy, [-1, 2])
            output_xy_grids.append(anchors_xy_flat)

            anchors_xy_centers = anchors_xy + 0.5 # centers
            anchors_xy_centers *= ratio
            ratios = tf.tile([ratio], [tf.shape(anchors_xy_flat)[0]])
            output_ratios.append(ratios)

            anchors_for_layer = tf.concat([anchors_xy_centers, anchors_wh], axis=-1)

            anchors_flat = tf.reshape(anchors_for_layer, [-1, 4])

            logger.info('base_scale: {}: output_size: {}, anchors_for_layer: {}, anchors_flat: {}'.format(base_scale, output_size, anchors_for_layer.shape, anchors_flat.shape))
            anchors_abs_coords.append(anchors_flat)

        anchors_all = tf.concat(anchors_abs_coords, axis=0)
        output_xy_grids = tf.concat(output_xy_grids, axis=0)
        output_ratios = tf.concat(output_ratios, axis=0)

        return anchors_all, output_xy_grids, output_ratios

def create_xy_grid(batch_size, output_size, anchors_per_scale):
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchors_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    return xy_grid

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

def generate_true_values_for_anchors(orig_bboxes, orig_texts, anchors_all, output_xy_grids, output_ratios, image_size):
    orig_bboxes = tf.convert_to_tensor(orig_bboxes, dtype=tf.float32)

    #num_examples = 9
    #orig_bboxes = tf.zeros([num_examples, 4], dtype=tf.float32)

    num_scales = 3
    num_boxes = 3

    ious = loss.box_iou(orig_bboxes, anchors_all)
    #logger.info('ious: {}, orig_bboxes: {}, anchors_all: {}, output_xy_grids: {}, output_ratios: {}'.format(
    #    ious.shape, orig_bboxes.shape, anchors_all.shape, output_xy_grids.shape, output_ratios.shape))

    #tf.print('bboxes:', orig_bboxes)

    best_anchors_index = tf.argmax(ious, 1)
    #tf.print('best_anchors_index:', best_anchors_index)
    num_anchors = anchors_all.shape[0]

    true_values_for_loss = tf.zeros([num_anchors, 4 + 1 + 1])
    true_values_abs = tf.zeros([num_anchors, 4 + 1 + 1])

    best_anchors = tf.gather(anchors_all, best_anchors_index) # [N, 4]
    best_xy_grids = tf.gather(output_xy_grids, best_anchors_index) # [N, 2]
    best_ratios = tf.gather(output_ratios, best_anchors_index)
    best_ratios = tf.expand_dims(best_ratios, 1) # [N, 1]

    #logger.info('best_anchors_index: {}, best_anchors: {}, best_xy_grids: {}, best_ratios: {}'.format(best_anchors_index.shape, best_anchors.shape, best_xy_grids.shape, best_ratios.shape))

    cx, cy, w, h = tf.split(orig_bboxes, num_or_size_splits=4, axis=1)

    _, _, aw, ah = tf.split(best_anchors, num_or_size_splits=4, axis=1)
    w_for_loss = tf.math.log(tf.maximum(w, 1) / aw)
    h_for_loss = tf.math.log(tf.maximum(h, 1) / ah)

    #tf.print('bboxes:', orig_bboxes, 'grid:', best_xy_grids, 'ratio:', best_ratios, 'best_anchors:', best_anchors)

    grid_x, grid_y = tf.split(best_xy_grids, num_or_size_splits=2, axis=1)
    # this is offset within a cell on the particular scale measured in relative to this scale units
    # this is what sigmoid(pred_xy) should be equal to
    # loss will use tf.nn.sigmoid_cross_entropy_with_logits() to match logits (Tx, Ty) to these true values, otherwise true value should've been logit() (aka reverse sigmoid)
    x_for_loss = cx / best_ratios - grid_x
    y_for_loss = cy / best_ratios - grid_y

    #tf.print('x_for_loss:', x_for_loss)
    #logger.info('cx: {}, grid_x: {}, x_for_loss: {}, aw: {}, w: {}, w_for_loss: {}'.format(cx.shape, grid_x.shape, x_for_loss.shape, aw.shape, w.shape, w_for_loss.shape))

    bboxes_for_loss = tf.concat([x_for_loss, y_for_loss, w_for_loss, h_for_loss], axis=1)
    objs_for_loss = tf.ones_like(x_for_loss)
    has_text_for_loss = tf.where(tf.not_equal(orig_texts, '<SKIP>'), 1, 0)
    values_for_loss = tf.concat([bboxes_for_loss, objs_for_loss, has_text_for_loss], axis=1)

    update_idx = tf.expand_dims(best_anchors_index, 1)

    logger.info('update true_values: bboxes: {}, objs: {}, values: {}, update_idx: {}'.format(
        bboxes_for_loss.shape, objs_for_loss.shape, values_for_loss.shape, update_idx.shape))
    output = tf.tensor_scatter_nd_update(true_values_for_loss, update_idx, values_for_loss)

    true_texts = tf.constant('<SKIP>')
    true_texts = tf.expand_dims(true_texts, 0)
    true_texts = tf.tile(true_texts, [num_anchors])
    true_texts = tf.expand_dims(true_texts, 1)

    update_labels = tf.expand_dims(orig_texts, 1)
    true_texts = tf.tensor_scatter_nd_update(true_texts, update_idx, update_labels)
    true_texts = tf.squeeze(true_texts, 1)

    return output, true_texts
