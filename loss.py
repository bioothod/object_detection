import tensorflow as tf
from tensorflow.python.ops import array_ops

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 1. #2
        self.reduction = reduction
        self.from_logits = from_logits

    def call(self, y_true, y_pred, weights=None):
        true_shape = y_true.shape
        pred_shape = y_pred.shape

        assert true_shape == pred_shape

        #y_true = tf.reshape(y_true, [-1, true_shape[-1]])
        #y_pred = tf.reshape(y_pred, [-1, pred_shape[-1]])

        y_true = tf.clip_by_value(y_true, 0, 1)

        if self.from_logits:
            sigmoid_p = tf.nn.sigmoid(y_pred)
        else:
            sigmoid_p = y_pred

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)

        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1)) - \
            (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1))

        if self.reduction==tf.keras.losses.Reduction.NONE:
            #return tf.reshape(per_entry_cross_ent, true_shape)
            return per_entry_cross_ent

        raise "qwe"
