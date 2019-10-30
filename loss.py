import logging

import tensorflow as tf

import yolo

logger = logging.getLogger('detection')

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
    true_box_xy = valid_true_boxes[..., 0:2]
    true_box_wh = valid_true_boxes[..., 2:4]

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


class YOLOLoss:
    def __init__(self,
                 anchors,
                 grid_xy,
                 ratios,
                 image_size,
                 num_classes,
                 obj_scale=1.,
                 noobj_scale=1.,
                 dist_scale=1.,
                 **kwargs):

        #super(YOLOLoss, self).__init__(**kwargs)

        self.image_size = image_size
        self.num_classes = num_classes

        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.dist_scale = dist_scale

        # added batch dimension
        self.grid_xy = tf.expand_dims(grid_xy, 0)
        self.ratios = tf.expand_dims(ratios, 0)
        self.ratios = tf.expand_dims(ratios, -1) # [B, N, 1]
        self.anchors_wh = tf.expand_dims(anchors[..., :2], 0)

    def gen_ignore_mask(self, input_tuple, object_mask, true_bboxes):
        idx, pred_boxes_for_single_image = input_tuple
        valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], tf.bool))
        iou = box_iou(pred_boxes_for_single_image, valid_true_boxes)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        #logger.info('pred_boxes_for_single_image: {}, valid_true_boxes: {}, iou: {}, best_iou: {}, ignore_mask_tmp: {}'.format(
        #    pred_boxes_for_single_image.shape, valid_true_boxes.shape, iou.shape, best_iou.shape, ignore_mask_tmp.shape))
        return ignore_mask_tmp

    def focal_loss(self, y_true, y_pred, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(y_true - y_pred), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou


    def call(self, y_true, y_pred):
        object_mask = tf.expand_dims(y_true[..., 4], -1)
        true_conf = object_mask

        #sigmoid(t_xy) + c_xy
        pred_box_xy = self.grid_xy + tf.sigmoid(y_pred[..., :2])
        pred_box_xy = pred_box_xy * self.ratios

        pred_wh = tf.clip_by_value(y_pred[..., 2:4], -7, 7)
        pred_box_wh = tf.math.exp(pred_wh) * self.anchors_wh

        # confidence/objectiveness
        pred_box_conf = tf.expand_dims(y_pred[..., 4], -1)
        pred_box_conf_prob = tf.sigmoid(pred_box_conf)

        true_classes = y_true[..., 5:]
        pred_classes = y_pred[..., 5:]


        true_xy = (y_true[..., 0:2] + self.grid_xy) * self.ratios
        true_wh = tf.math.exp(y_true[..., 2:4]) * self.anchors_wh

        pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
        true_bboxes = tf.concat([true_xy, true_wh], axis=-1)



        wh_scale = true_wh / self.image_size
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=-1)

        giou = tf.expand_dims(self.bbox_giou(pred_bboxes, true_bboxes), axis=-1)
        dist_loss = 1 - giou
        dist_loss = dist_loss * wh_scale * object_mask
        dist_loss = tf.reduce_sum(dist_loss, [1, 2])


        smooth_true_conf = true_conf
        if True:
            delta = 0.04
            smooth_true_conf = (1 - delta) * true_conf + delta / 2 # 2 - is number of classes for objectness



        ignore_mask = tf.map_fn(lambda t: self.gen_ignore_mask(t, object_mask, true_bboxes),
                            (tf.range(tf.shape(pred_bboxes)[0]), pred_bboxes),
                            parallel_iterations=32,
                            back_prop=True,
                            dtype=(tf.float32))
        # shape: [B, N, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)


        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_conf, logits=pred_box_conf)

        focal_conf_loss = self.focal_loss(y_true=true_conf, y_pred=pred_box_conf_prob)
        conf_loss *= focal_conf_loss

        conf_loss_pos = object_mask * conf_loss
        conf_loss_pos = tf.reduce_sum(conf_loss_pos, [1, 2])

        conf_loss_neg = (1 - object_mask) * conf_loss * ignore_mask
        conf_loss_neg = tf.reduce_sum(conf_loss_neg, [1, 2])


        total_loss = dist_loss * self.dist_scale, conf_loss_pos * self.obj_scale, conf_loss_neg * self.noobj_scale
        return total_loss
