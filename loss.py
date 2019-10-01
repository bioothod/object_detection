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

def gen_ignore_mask(input_tuple, object_mask, true_bboxes):
    idx, pred_boxes_for_single_image = input_tuple
    valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], tf.bool))
    iou = box_iou(pred_boxes_for_single_image, valid_true_boxes)
    best_iou = tf.reduce_max(iou, axis=-1)
    ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
    #logger.info('pred_boxes_for_single_image: {}, valid_true_boxes: {}, iou: {}, best_iou: {}, ignore_mask_tmp: {}'.format(
    #    pred_boxes_for_single_image.shape, valid_true_boxes.shape, iou.shape, best_iou.shape, ignore_mask_tmp.shape))
    return ignore_mask_tmp

class YOLOLoss:
    def __init__(self,
                 anchors,
                 grid_xy,
                 ratios,
                 image_size,
                 obj_scale=5.,
                 noobj_scale=4e-3,
                 dist_scale=2.,
                 class_scale=1.,
                 **kwargs):

        #super(YOLOLoss, self).__init__(**kwargs)

        self.image_size = image_size

        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.dist_scale = dist_scale
        self.class_scale = class_scale

        # added batch dimension
        self.grid_xy = tf.expand_dims(grid_xy, 0)
        self.ratios = tf.expand_dims(ratios, 0)
        self.ratios = tf.expand_dims(ratios, -1) # [B, N, 1]
        self.anchors_wh = tf.expand_dims(anchors[..., :2], 0)

    def call(self, y_true, y_pred):
        object_mask = tf.expand_dims(y_true[..., 4], -1)
        true_conf = object_mask

        #sigmoid(t_xy) + c_xy
        pred_box_xy = self.grid_xy + tf.sigmoid(y_pred[..., :2])
        pred_box_xy = pred_box_xy * self.ratios

        pred_box_wh = tf.math.exp(y_pred[..., 2:4]) * self.anchors_wh

        # confidence/objectiveness
        pred_box_conf = tf.expand_dims(y_pred[..., 4], -1)

        true_classes = y_true[..., 5:]
        pred_classes = y_pred[..., 5:]


        true_xy = (tf.sigmoid(y_true[..., 0:2]) + self.grid_xy) * self.ratios
        true_wh = tf.math.exp(y_true[..., 2:4]) * self.anchors_wh

        pred_bboxes = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
        true_bboxes = tf.concat([true_xy, true_wh], axis=-1)

        ignore_mask = tf.map_fn(lambda t: gen_ignore_mask(t, object_mask, true_bboxes),
                            (tf.range(tf.shape(pred_bboxes)[0]), pred_bboxes),
                            parallel_iterations=32,
                            back_prop=True,
                            dtype=(tf.float32))
        # shape: [B, N, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)


        wh_scale = true_wh / self.image_size
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=-1)

        l2_loss = tf.nn.l2_loss(y_pred[..., :2]) + tf.nn.l2_loss(y_pred[..., 4]) + tf.nn.l2_loss(y_pred[..., 5:])

        dist_loss_xy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., :2], logits=y_pred[..., :2])
        dist_loss_wh = tf.square(y_true[..., 2:4] - y_pred[..., 2:4])
        dist_loss = dist_loss_xy + dist_loss_wh
        dist_loss = dist_loss * wh_scale * object_mask
        dist_loss = tf.reduce_sum(dist_loss, [1, 2])

        smooth_true_classes = true_classes
        if True:
            label_smoothing = 0.1
            smooth_true_classes = true_classes * (1.0 - label_smoothing) + 0.5 * label_smoothing

        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_classes, logits=pred_classes)

        nan_obj = tf.math.is_nan(class_loss)
        num_nans = tf.math.count_nonzero(nan_obj)
        if num_nans != 0:
            tf.print('nans in class_loss1:', num_nans, 'class_loss:', tf.shape(class_loss), 'total nodes:', tf.math.reduce_prod(tf.shape(class_loss)))

        class_loss = tf.reduce_sum(class_loss, -1)
        class_loss = tf.expand_dims(class_loss, -1)

        nan_obj = tf.math.is_nan(class_loss)
        num_nans = tf.math.count_nonzero(nan_obj)
        if num_nans != 0:
            tf.print('nans in class_loss2:', num_nans, 'class_loss:', tf.shape(class_loss), 'total nodes:', tf.math.reduce_prod(tf.shape(class_loss)))

        class_loss1 = object_mask * class_loss

        nan_obj = tf.math.is_nan(class_loss)
        num_nans = tf.math.count_nonzero(nan_obj)
        if num_nans != 0:
            tf.print('nans in class_loss3:', num_nans, 'class_loss:', tf.shape(class_loss), 'total nodes:', tf.math.reduce_prod(tf.shape(class_loss)))
        nan_obj = tf.math.is_nan(class_loss1)
        num_nans = tf.math.count_nonzero(nan_obj)
        if num_nans != 0:
            tf.print('nans in class_loss4:', num_nans, 'class_loss:', tf.shape(class_loss), 'total nodes:', tf.math.reduce_prod(tf.shape(class_loss)))

        nan_obj = tf.math.is_nan(object_mask)
        num_nans = tf.math.count_nonzero(nan_obj)
        if num_nans != 0:
            tf.print('nans in object_mask3:', num_nans, 'object_mask:', tf.shape(object_mask), 'total nodes:', tf.math.reduce_prod(tf.shape(object_mask)))


        class_loss = tf.reduce_sum(class_loss1, [1, 2])


        smooth_true_conf = true_conf
        if True:
            label_smoothing = 0.1
            smooth_true_conf = true_conf * (1.0 - label_smoothing) + 0.5 * label_smoothing
        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_conf, logits=pred_box_conf)


        conf_loss_pos = object_mask * conf_loss
        conf_loss_pos = tf.reduce_sum(conf_loss_pos, [1, 2])

        conf_loss_neg = (1 - object_mask) * conf_loss * ignore_mask
        conf_loss_neg = tf.reduce_sum(conf_loss_neg, [1, 2])

        return dist_loss * self.dist_scale, class_loss * self.class_scale, conf_loss_pos * self.obj_scale, conf_loss_neg * self.noobj_scale + l2_loss * 1e-2
