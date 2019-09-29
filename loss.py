import logging

import tensorflow as tf

import anchors_gen

logger = logging.getLogger('detection')

def _create_mesh_xy(batch_size, grid_h, grid_w, n_box):
    mesh_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    mesh_y = tf.transpose(mesh_x, (0, 2, 1, 3, 4))
    mesh_xy = tf.tile(tf.concat([mesh_x, mesh_y],-1), [batch_size, 1, 1, n_box, 1])

    return mesh_xy

def _create_mesh_anchor(anchors, batch_size, grid_h, grid_w, n_box):
    """
    # Returns
        mesh_xy : Tensor, shape of (batch_size, grid_h, grid_w, n_box, 2)
            [..., 0] means "anchor_w"
            [..., 1] means "anchor_h"
    """
    mesh_anchor = tf.tile(anchors, [batch_size*grid_h*grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_box, 2])
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor

def box_iou(pred_boxes, valid_true_boxes):
    '''
    param:
        pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        valid_true: [V, 4]
    '''

    # [13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou

def gen_ignore_mask(input_tuple, object_mask, true_bboxes):
    idx, pred_boxes_for_single_image = input_tuple
    # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
    # V: num of true gt box of each image in a batch
    valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
    # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
    iou = box_iou(pred_boxes_for_single_image, valid_true_boxes)
    # shape: [13, 13, 3]
    best_iou = tf.reduce_max(iou, axis=-1)
    # shape: [13, 13, 3]
    ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
    #logger.info('pred_boxes_for_single_image: {}, valid_true_boxes: {}, iou: {}, best_iou: {}, ignore_mask_tmp: {}'.format(
    #    pred_boxes_for_single_image.shape, valid_true_boxes.shape, iou.shape, best_iou.shape, ignore_mask_tmp.shape))
    return ignore_mask_tmp

class LossTensorCalculator:
    def __init__(self,
                 image_size=288, 
                 ignore_thresh=0.5, 
                 obj_scale=5.,
                 noobj_scale=1.,
                 dist_scale=1.,
                 class_scale=1.):
        self.ignore_thresh = ignore_thresh
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.dist_scale = dist_scale
        self.class_scale = class_scale        
        self.image_size = image_size

        self.bce = tf.keras.losses.BinaryCrossentropy()

    def run(self, y_true, y_pred, anchors_wh, output_size):
        ratio = self.image_size / float(output_size)

        pred_orig_shape = tf.shape(y_pred)
        true_orig_shape = tf.shape(y_true)

        y_true = tf.reshape(y_true, [-1, output_size, output_size, true_orig_shape[2], true_orig_shape[3]])
        y_pred = tf.reshape(y_pred, [-1, output_size, output_size, true_orig_shape[2], true_orig_shape[3]])


        object_mask = tf.expand_dims(y_true[..., 4], 4)
        true_conf = object_mask

        grid_offset = _create_mesh_xy(*y_pred.shape[:4])
        anchor_grid = _create_mesh_anchor(anchors_wh, *y_pred.shape[:4])


        #sigmoid(t_xy) + c_xy
        pred_box_xy = grid_offset + tf.sigmoid(y_pred[..., :2])
        pred_box_xy = pred_box_xy * ratio

        #anchors = tf.constant(anchors_wh, dtype=tf.float32, shape=[1, 1, 1, 3, 2])
        anchors_wh = tf.reshape(anchors_wh, [3, 2])
        pred_box_wh = tf.math.exp(y_pred[..., 2:4]) * anchors_wh

        # confidence/objectiveness
        pred_box_conf = tf.expand_dims(y_pred[..., 4], -1)

        true_classes = y_true[..., 5:]
        pred_classes = y_pred[..., 5:]


        true_xy = y_true[..., 0:2] * ratio
        true_wh = tf.math.exp(y_true[..., 2:4]) * anchor_grid

        pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
        true_bboxes = tf.concat([true_xy, true_wh], axis=-1)

        ignore_mask = tf.map_fn(lambda t: gen_ignore_mask(t, object_mask, true_bboxes),
                            (tf.range(tf.shape(pred_bboxes)[0]), pred_bboxes),
                            parallel_iterations=32,
                            back_prop=True,
                            dtype=(tf.float32))

        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        wh_scale = true_wh / self.image_size
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) 

        dist_loss = tf.square(y_pred[..., :4] - y_true[..., :4])
        dist_loss = dist_loss * wh_scale * object_mask
        dist_loss = tf.reduce_sum(dist_loss, list(range(1, 5)))

        smooth_true_classes = true_classes
        if True:
            label_smoothing = 0.1
            smooth_true_classes = true_classes * (1.0 - label_smoothing) + 0.5 * label_smoothing
        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_classes, logits=pred_classes)
        class_loss = tf.reduce_sum(class_loss, -1)
        class_loss = tf.expand_dims(class_loss, -1)
        class_loss = object_mask * class_loss
        class_loss = tf.reduce_sum(class_loss, list(range(1, 5)))

        smooth_true_conf = true_conf
        if True:
            label_smoothing = 0.1
            smooth_true_conf = true_conf * (1.0 - label_smoothing) + 0.5 * label_smoothing
        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_conf, logits=pred_box_conf)

        conf_loss_pos = object_mask * conf_loss
        conf_loss_pos = tf.reduce_sum(conf_loss_pos, list(range(1, 5)))

        conf_loss_neg = (1 - object_mask) * conf_loss * ignore_mask
        conf_loss_neg = tf.reduce_sum(conf_loss_neg, list(range(1, 5)))

        return dist_loss * self.dist_scale, class_loss * self.class_scale, conf_loss_pos * self.obj_scale, conf_loss_neg * self.noobj_scale

class YOLOLoss:
    def __init__(self, image_size, anchors, output_sizes, ignore_thresh=0.5, obj_scale=4, noobj_scale=0.001, dist_scale=2, class_scale=1, reduction=tf.keras.losses.Reduction.NONE):
        super(YOLOLoss, self).__init__()
        self.reduction = reduction
        self.anchors = anchors.reshape([len(output_sizes), -1])
        self.output_sizes = output_sizes

        self.calc = LossTensorCalculator(image_size=image_size,
                                        ignore_thresh=ignore_thresh, 
                                        obj_scale=obj_scale,
                                        noobj_scale=noobj_scale,
                                        dist_scale=dist_scale,
                                        class_scale=class_scale)

    def call(self, y_true_list, y_pred_list):
        dist_loss = tf.zeros((tf.shape(y_true_list[0])[0]), dtype=tf.float32)
        class_loss = tf.zeros((tf.shape(y_true_list[0])[0]), dtype=tf.float32)
        conf_loss_pos = tf.zeros((tf.shape(y_true_list[0])[0]), dtype=tf.float32)
        conf_loss_neg = tf.zeros((tf.shape(y_true_list[0])[0]), dtype=tf.float32)

        for output_size, anchors, y_true, y_pred in zip(self.output_sizes, self.anchors, y_true_list, y_pred_list):
            dist_loss_, class_loss_, conf_loss_pos_, conf_loss_neg_ = self.calc.run(y_true, y_pred, anchors, output_size)
            dist_loss += dist_loss_
            class_loss += class_loss_
            conf_loss_pos += conf_loss_pos_
            conf_loss_neg += conf_loss_neg_

        return dist_loss, class_loss, conf_loss_pos, conf_loss_neg
