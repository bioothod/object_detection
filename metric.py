import logging

import tensorflow as tf

import anchors

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

class Metric:
    def __init__(self, training=True, **kwargs):
        self.training = training

        self.total_loss = tf.keras.metrics.Mean()
        self.reg_loss = tf.keras.metrics.Mean()
        self.dist_loss = tf.keras.metrics.Mean()
        self.ce_loss = tf.keras.metrics.Mean()
        self.ce_image_loss = tf.keras.metrics.Mean()

        self.ce_acc = tf.keras.metrics.CategoricalAccuracy()
        self.ce_image_acc = tf.keras.metrics.CategoricalAccuracy()

    def reset_states(self):
        self.total_loss.reset_states()
        self.reg_loss.reset_states()
        self.dist_loss.reset_states()
        self.ce_loss.reset_states()
        self.ce_image_loss.reset_states()

        self.ce_acc.reset_states()
        self.ce_image_acc.reset_states()

    def str_result(self):
        return 'total_loss: {:.4f}, reg_loss: {:.3f}, dist: {:.3f}, ce: {:.3f}, image_ce: {:.3f}, acc: {:.3f}, image_acc: {:.3f}'.format(
                self.total_loss.result(),
                self.reg_loss.result(),
                self.dist_loss.result(),
                self.ce_loss.result(),
                self.ce_image_loss.result(),

                self.ce_acc.result(),
                self.ce_image_acc.result(),
                )


class ModelMetric:
    def __init__(self,
                 all_anchors: tf.Tensor,
                 num_classes: int,
                 dtype: tf.dtypes.DType = tf.float32,
                 **kwargs):

        self.all_anchors = all_anchors
        self.num_classes = num_classes
        self.dtype = dtype

        self.train_metric = Metric(training=True, name='train_metric')

        self.dist_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def str_result(self, training):
        return self.train_metric.str_result()

    def reset_states(self):
        self.train_metric.reset_states()

    def __call__(self, images, true_bboxes, true_labels, true_image_labels, pred_bboxes, pred_scores, pred_image_scores, training):
        true_bboxes, true_labels = anchors.anchor_targets_bbox(self.all_anchors, images, true_bboxes, true_labels, self.num_classes, dtype=self.dtype)

        y_shape = tf.shape(true_labels)
        batch = y_shape[0]
        n_anchors = y_shape[1]

        anchors_states = true_labels[:, :, -1]
        not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)

        true_not_ignore_labels = tf.gather_nd(true_labels[:, :, :-1], not_ignore_idx)
        pred_not_ignore_scores = tf.gather_nd(pred_scores, not_ignore_idx)

        true_bboxes = tf.gather_nd(true_bboxes[:, :, :-1], true_idx)
        pred_bboxes = tf.gather_nd(pred_bboxes, true_idx)

        dist_loss = self.dist_loss(true_bboxes, pred_bboxes)
        class_loss = focal_loss(true_not_ignore_labels, pred_not_ignore_scores, reduction='sum')

        # normalizer equals zero when there are no bboxes and there are only image classes
        if normalizer != 0:
            dist_loss = tf.divide(dist_loss, normalizer)
            class_loss = tf.divide(class_loss, normalizer)

        self.train_metric.dist_loss.update_state(dist_loss)
        self.train_metric.ce_loss.update_state(class_loss)

        true_fg_labels = tf.gather_nd(true_labels[:, :, :-1], true_idx)
        pred_fg_scores = tf.gather_nd(pred_scores, true_idx)
        self.train_metric.ce_acc.update_state(true_fg_labels, pred_fg_scores)

        image_class_loss = 0.
        if pred_image_scores is not None:
            image_class_loss = focal_loss(true_image_labels, pred_image_scores, reduction='sum')
            self.train_metric.ce_image_loss.update_state(image_class_loss)
            self.train_metric.ce_image_acc.update_state(true_image_labels, pred_image_scores)

        return dist_loss, class_loss, image_class_loss
