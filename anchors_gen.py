import logging

import numpy as np
import tensorflow as tf

import loss
import yolo

logger = logging.getLogger('detection')

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

def generate_true_labels_for_anchors(orig_bboxes, orig_labels, anchors_all, output_xy_grids, output_ratios, image_size, num_classes):
    orig_bboxes = tf.convert_to_tensor(orig_bboxes, dtype=tf.float32)
    orig_labels = tf.convert_to_tensor(orig_labels, dtype=tf.int32)

    #num_examples = 9
    #orig_bboxes = tf.zeros([num_examples, 4], dtype=tf.float32)
    #orig_labels = tf.ones([num_examples,], dtype=tf.int32)

    num_scales = 3
    num_boxes = 3

    scaled_size = image_size / yolo.DOWNSAMPLE_RATIO

    ious = loss.box_iou(orig_bboxes, anchors_all)
    #logger.info('ious: {}, orig_bboxes: {}, anchors_all: {}, output_xy_grids: {}, output_ratios: {}'.format(
    #    ious.shape, orig_bboxes.shape, anchors_all.shape, output_xy_grids.shape, output_ratios.shape))

    #tf.print('bboxes:', orig_bboxes)

    best_anchors_index = tf.argmax(ious, 1)
    #tf.print('best_anchors_index:', best_anchors_index)
    num_anchors = anchors_all.shape[0]

    true_values_for_loss = tf.zeros([num_anchors, 4 + 1 + num_classes])
    true_values_abs = tf.zeros([num_anchors, 4 + 1 + num_classes])

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
    labels_for_loss = tf.one_hot(orig_labels, num_classes, dtype=tf.float32)
    values_for_loss = tf.concat([bboxes_for_loss, objs_for_loss, labels_for_loss], axis=1)

    update_idx = tf.expand_dims(best_anchors_index, 1)

    logger.info('update true_values: bboxes: {}, objs: {}, labels: {}, values: {}, update_idx: {}'.format(
        bboxes_for_loss.shape, objs_for_loss.shape, labels_for_loss.shape, values_for_loss.shape, update_idx.shape))
    output = tf.tensor_scatter_nd_update(true_values_for_loss, update_idx, values_for_loss)

    return output
