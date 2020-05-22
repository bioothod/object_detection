import logging

import tensorflow as tf

logger = logging.getLogger('detection')

def focal_loss(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               gamma: float = 1.5,
               alpha: float = 0.25,
               from_logits: bool = False,
               reduction: str = 'sum'):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if from_logits:
        y_pred = tf.nn.softmax(y_pred)

    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha = tf.ones_like(y_true) * alpha
    alpha = tf.where(tf.equal(y_true, 1.), alpha, 1 - alpha)

    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1 - y_pred)

    loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
    loss = tf.reduce_sum(loss, axis=-1)

    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss

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

class Metric:
    def __init__(self, training=True, **kwargs):
        self.training = training

        # these will be updated in the training code
        self.total_loss = tf.keras.metrics.Mean(name='total_loss')
        self.reg_loss = tf.keras.metrics.Mean(name='reg_loss')

        self.dist_loss = tf.keras.metrics.Mean(name='dist_loss')
        self.class_loss = tf.keras.metrics.Mean(name='class_loss')
        self.conf_loss_pos = tf.keras.metrics.Mean(name='conf_loss_pos')
        self.conf_loss_neg = tf.keras.metrics.Mean(name='conf_loss_neg')

        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        self.obj_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='objectness_accuracy')

        self.best_ious_for_true_bboxes_metric = tf.keras.metrics.Mean(name='best_ious')
        self.mean_ious_for_true_bboxes_metric = tf.keras.metrics.Mean(name='mean_ious')
        self.num_good_ious_metric = tf.keras.metrics.Mean(name='num_good_ious')

        self.num_positive_labels_metric = tf.keras.metrics.Mean(name='num_positive_labels_ious')

    def reset_states(self):
        self.total_loss.reset_states()
        self.reg_loss.reset_states()

        self.dist_loss.reset_states()
        self.class_loss.reset_states()
        self.conf_loss_pos.reset_states()
        self.conf_loss_neg.reset_states()

        self.accuracy_metric.reset_states()
        self.obj_accuracy_metric.reset_states()

        self.best_ious_for_true_bboxes_metric.reset_states()
        self.mean_ious_for_true_bboxes_metric.reset_states()

        self.num_good_ious_metric.reset_states()
        self.num_positive_labels_metric.reset_states()

    def str_result(self):
        return 'total_loss: {:.3f}, reg_loss: {:.3f}, dist: {:.3f}, class: {:.3f}, conf: {:.3f}/{:.3f}, m_acc: {:.3f}, m_obj_acc: {:.3f}, num_ious: {:.1f}/{:.1f}, best/mean_ious: {:.3f}/{:.3f}'.format(
                self.total_loss.result(),
                self.reg_loss.result(),
                self.dist_loss.result(),
                self.class_loss.result(),

                self.conf_loss_pos.result(),
                self.conf_loss_neg.result(),

                self.accuracy_metric.result(),
                self.obj_accuracy_metric.result(),

                self.num_good_ious_metric.result(),
                self.num_positive_labels_metric.result(),

                self.best_ious_for_true_bboxes_metric.result(),
                self.mean_ious_for_true_bboxes_metric.result(),
                )


class ModelMetric:
    def __init__(self,
                 anchors_all: tf.Tensor,
                 output_xy_grids: tf.Tensor,
                 output_ratios: tf.Tensor,
                 image_size: int,
                 num_classes: int,
                 **kwargs):

        self.num_classes = num_classes
        self.image_size = image_size

        self.grid_xy = tf.expand_dims(output_xy_grids, 0)
        self.ratios = tf.expand_dims(output_ratios, 0)
        self.ratios = tf.expand_dims(self.ratios, -1) # [B, N, 1]
        self.anchors_wh = tf.expand_dims(anchors_all[..., :2], 0)

        self.train_metric = Metric(training=True, name='train')
        self.eval_metric = Metric(training=False, name='eval')

        self.obj_scale = 1
        self.noobj_scale = 1
        self.dist_scale = 1
        self.class_scale = 1

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def gen_ignore_mask(self, input_tuple, object_mask, true_bboxes, m):
        idx, pred_boxes_for_single_image = input_tuple
        valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], tf.bool))
        ious = box_iou(pred_boxes_for_single_image, valid_true_boxes)

        # ious: [N, V] shape
        # N - number of anchors
        # V - number of objects in the image, number of the true bounding boxes

        best_ious_for_anchors = tf.reduce_max(ious, axis=1)

        best_ious_for_true_bboxes = tf.reduce_max(ious, axis=0)
        mean_ious_for_true_bboxes = tf.reduce_mean(ious, axis=0)
        good_ious_num = tf.where(best_ious_for_true_bboxes > 0.5)
        m.best_ious_for_true_bboxes_metric.update_state(best_ious_for_true_bboxes)
        m.mean_ious_for_true_bboxes_metric.update_state(mean_ious_for_true_bboxes)
        m.num_good_ious_metric.update_state(tf.shape(good_ious_num)[0])

        ignore_mask = tf.cast(best_ious_for_anchors < 0.5, tf.float32)
        #logger.info('pred_boxes_for_single_image: {}, valid_true_boxes: {}, iou: {}, best_iou: {}, ignore_mask_tmp: {}'.format(
        #    pred_boxes_for_single_image.shape, valid_true_boxes.shape, iou.shape, best_iou.shape, ignore_mask_tmp.shape))
        # shape: [B, N, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        return ignore_mask

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

    def focal_loss(self, y_true, y_pred, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(y_true - y_pred), gamma)
        return focal_loss

    def __call__(self, y_true, y_pred, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        object_mask = tf.expand_dims(y_true[..., 4], -1)
        object_mask_bool = tf.cast(object_mask, 'bool')

        num_objects = tf.cast(tf.math.count_nonzero(object_mask), tf.int32)
        m.num_positive_labels_metric.update_state(num_objects)

        true_conf = object_mask

        #sigmoid(t_xy) + c_xy
        pred_box_xy = self.grid_xy + tf.sigmoid(y_pred[..., :2])
        pred_box_xy = pred_box_xy * self.ratios

        pred_wh = tf.clip_by_value(y_pred[..., 2:4], -7, 7)
        pred_box_wh = tf.math.exp(pred_wh) * self.anchors_wh

        # confidence/objectiveness
        true_obj_masked = tf.boolean_mask(object_mask, object_mask_bool)
        pred_box_conf = tf.expand_dims(y_pred[..., 4], -1)
        pred_box_conf_prob = tf.sigmoid(pred_box_conf)
        pred_box_conf_prob_true_masked = tf.boolean_mask(pred_box_conf_prob, object_mask_bool)
        m.obj_accuracy_metric.update_state(y_true=true_obj_masked, y_pred=pred_box_conf_prob_true_masked)

        true_classes = y_true[..., 5:]
        pred_classes = y_pred[..., 5:]
        true_classes_masked = tf.boolean_mask(true_classes, object_mask_bool)
        pred_classes_masked = tf.boolean_mask(pred_classes, object_mask_bool)
        m.accuracy_metric.update_state(y_true=true_classes_masked, y_pred=pred_classes_masked)


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
        m.dist_loss.update_state(dist_loss)


        smooth_true_classes = true_classes
        if True:
            delta = 0.04
            smooth_true_classes = (1 - delta) * true_classes + delta / self.num_classes
        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_classes, logits=pred_classes)
        class_loss = tf.reduce_sum(class_loss, -1)
        class_loss = tf.expand_dims(class_loss, -1)

        class_loss = object_mask * class_loss
        class_loss = tf.reduce_sum(class_loss, [1, 2])
        m.class_loss.update_state(class_loss)

        ignore_mask = tf.map_fn(lambda t: self.gen_ignore_mask(t, object_mask, true_bboxes, m),
                            (tf.range(tf.shape(pred_bboxes)[0]), pred_bboxes),
                            parallel_iterations=32,
                            back_prop=True,
                            dtype=(tf.float32))

        smooth_true_conf = true_conf
        if True:
            delta = 0.04
            smooth_true_conf = (1 - delta) * true_conf + delta / 2 # 2 - is number of classes for objectness

        conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=smooth_true_conf, logits=pred_box_conf)
        focal_conf_loss = self.focal_loss(y_true=true_conf, y_pred=pred_box_conf_prob)
        conf_loss *= focal_conf_loss

        conf_loss_pos = object_mask * conf_loss
        conf_loss_pos = tf.reduce_sum(conf_loss_pos, [1, 2])
        m.conf_loss_pos.update_state(conf_loss_pos)

        conf_loss_neg = (1 - object_mask) * conf_loss * ignore_mask
        conf_loss_neg = tf.reduce_sum(conf_loss_neg, [1, 2])
        m.conf_loss_neg.update_state(conf_loss_neg)

        total_loss = dist_loss * self.dist_scale + class_loss * self.class_scale + conf_loss_pos * self.obj_scale + conf_loss_neg * self.noobj_scale
        total_loss = tf.reduce_mean(total_loss)
        return total_loss
