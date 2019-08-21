import tensorflow as tf
from tensorflow.python.ops import array_ops

class CategoricalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE, class_weights=1.):
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
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1-1e-10)

        per_entry_ce = -y_true * tf.math.log(y_pred) * self.class_weights

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_ce

        raise "qwe"

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = kwargs['backend']
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x

def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs

def get_reduce_axes(per_image, data_format):
    axes = [1, 2] if data_format == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes

def round_if_needed(x, threshold):
    if threshold is not None:
        dtype = x.dtype
        x = tf.greater(x, threshold)
        x = tf.cast(x, dtype)

    return x

def average(x, per_image=False, class_weights=None, keepdims=True):
    if per_image:
        x = tf.reduce_mean(x, axis=0, keepdims=keepdims)
    if class_weights is not None:
        x = x * class_weights
    return tf.reduce_mean(x, keepdims=keepdims)

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE, beta=1, class_weights=1.):
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.data_format = 'channels_last'
        self.reduction = reduction
        self.class_weights = class_weights
        self.beta = beta
        self.smooth = 1e-8
        self.per_image = False
        self.threshold = None

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = round_if_needed(y_pred, self.threshold)
        axes = get_reduce_axes(self.per_image, self.data_format)

        # calculate score
        tp = tf.reduce_sum(y_true * y_pred, axis=axes, keepdims=True)
        fp = tf.reduce_sum(y_pred, axis=axes, keepdims=True) - tp
        fn = tf.reduce_sum(y_true, axis=axes, keepdims=True) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return score

        score = average(score, self.per_image, self.class_weights)

        return 1. - score

class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, reduction=tf.keras.losses.Reduction.NONE):
        super(CategoricalFocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2
        self.reduction = reduction
        self.from_logits = from_logits
        self.data_format = 'channels_last'

    def call(self, y_true, y_pred):
        true_shape = y_true.shape
        pred_shape = y_pred.shape

        assert true_shape == pred_shape

        if self.from_logits:
            axis = 3 if self.data_format() == 'channels_last' else 1
            y_pred /= tf.math.sum(y_pred, axis=axis, keepdims=True)

        y_true = tf.cast(y_true, y_pred.dtype)
        #y_true = tf.clip_by_value(y_true, 1e-10, 1-1e-10)
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1-1e-10)

        per_entry_ce = -y_true * (self.alpha * tf.math.pow((1 - y_pred), self.gamma) * tf.math.log(y_pred))

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_ce

        raise "qwe"

class IOUScore(tf.keras.metrics.Metric):
    def __init__(self,
            class_weights=1.,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=1e-8,
            name=None,
            **kwargs):
        super(IOUScore, self).__init__(name=name, **kwargs)

        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth
        self.data_format = 'channels_last'

        self.iou_score = self.add_weight(name='iou_score', initializer='zeros')
        self.iou_num = self.add_weight(name='iou_num', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = round_if_needed(y_pred, self.threshold)
        axes = get_reduce_axes(self.per_image, self.data_format)

        # score calculation
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true + y_pred, axis=axes) - intersection

        score = (intersection + self.smooth) / (union + self.smooth)
        score = tf.reduce_mean(score)
        #score = average(score, self.per_image, self.class_weights, keepdims=False)

        self.iou_score.assign_add(score)
        self.iou_num.assign_add(1)

    def result(self):
        return (self.iou_score + self.smooth) / (self.iou_num + self.smooth)
