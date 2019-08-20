import tensorflow as tf
from tensorflow.python.ops import array_ops

class CategoricalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE, class_weights=None):
        super(CategoricalLoss, self).__init__()
        self.from_logits = from_logits
        self.data_format = 'channels_last'
        self.reduction = reduction
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        true_shape = y_true.shape
        pred_shape = y_pred.shape

        assert true_shape == pred_shape

        if self.from_logits:
            axis = 3 if self.data_format() == 'channels_last' else 1
            y_pred /= tf.math.sum(y_pred, axis=axis, keepdims=True)

        #y_true = tf.clip_by_value(y_true, 1e-10, 1-1e-10)
        y_pred = tf.clip_by_value(y_true, 1e-10, 1-1e-10)

        per_entry_ce = -y_true * tf.math.log(y_pred) * self.class_weights

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_ce

        raise "qwe"

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 1. #2
        self.reduction = reduction
        self.from_logits = from_logits
        self.data_format = 'channels_last'

    def call(self, y_true, y_pred, weights=None):
        true_shape = y_true.shape
        pred_shape = y_pred.shape

        assert true_shape == pred_shape
        assert y_true.dtype == y_pred.dtype
        assert y_pred.dtype == tf.float32

        #y_true = tf.reshape(y_true, [-1, true_shape[-1]])
        #y_pred = tf.reshape(y_pred, [-1, pred_shape[-1]])

        if self.from_logits:
            axis = 3 if self.data_format() == 'channels_last' else 1
            y_pred /= tf.math.sum(y_pred, axis=axis, keepdims=True)

        #y_true = tf.clip_by_value(y_true, 1e-10, 1-1e-10)
        #y_pred = tf.clip_by_value(y_true, 1e-10, 1-1e-10)

        per_entry_ce = -y_true * (self.alpha * tf.math.pow(1 - y_pred), self.gamma) * tf.math.log(y_pred)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_ce

        raise "qwe"
