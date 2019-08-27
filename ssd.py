import collections
import logging

import numpy as np
import tensorflow as tf

import efficientnet.tfkeras as efn

logger = logging.getLogger('objdet')
logger.setLevel(logging.INFO)

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

class StdConv(tf.keras.layers.Layer):
    def __init__(self, global_params, num_filters, strides=2, dropout_rate=0.1):
        super(StdConv, self).__init__()

        self.num_filters = num_filters
        self.strides = strides
        self.dropout_rate = dropout_rate

        self.batch_norm_momentum = global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.data_format = global_params.data_format
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_dims = [2, 3]
        else:
            self.channel_axis = -1
            self.spatial_dims = [1, 2]

        self._build()

    def _build(self):
        self.c0 = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=3,
            strides=self.strides,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self.bn0 = tf.keras.layers.BatchNormalization(
            axis=self.channel_axis,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon)

        self.dropout0 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=True):
        return self.dropout0(self.bn0(self.c0(inputs), training=training))

def flaten_conv(x, k):
    return tf.reshape(x, (-1, tf.shape(x)[-1] // k))

class OutConv(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes):
        super(OutConv, self).__init__()

        self.k = k
        self.num_classes = num_classes

        self.data_format = global_params.data_format

        self._build()

    def _build(self):
        self.class_out = tf.keras.layers.Conv2D(
            filters=self.num_classes*self.k,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self.loc_out = tf.keras.layers.Conv2D(
            filters=4*self.k,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

    def call(self, inputs, training=True):
        return self.loc_out(inputs), self.class_out(inputs)

class SSD_Head(tf.keras.models.Model):
    def __init__(self, global_params, k, num_classes):
        super(SSD_Head, self).__init__()

        self.k = k
        self.num_classes = num_classes

        self.global_params = global_params
        self.relu_fn = global_params.relu_fn or tf.nn.swish

        self._build()

    def _build(self):
        self.dropout = tf.keras.layers.Dropout(0.25)
        self.sc0 = StdConv(self.global_params, 256, strides=1)
        self.sc1 = StdConv(self.global_params, 256)
        self.out = OutConv(self.global_params, self.k, self.num_classes)

    def call(self, inputs, training=True):
        x = self.relu_fn(inputs)
        x = self.dropout(x)
        x = self.sc0(x)
        x = self.sc1(x)
        x = self.out(x)

        return x

class SSD_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name=None):
        super(SSD_Loss, self).__init__(reduction, name)

    def __call__(y_true, y_pred, sample_weight=None):
        logger.info('true: {}, pred: {}'.format(y_true.shape, y_pred.shape))

def create_base_model(dtype, model_name):
    model_map = {
        'efficientnet-b0': (efn.EfficientNetB0, 224),
        'efficientnet-b1': (efn.EfficientNetB1, 240),
        'efficientnet-b2': (efn.EfficientNetB2, 260),
        'efficientnet-b3': (efn.EfficientNetB3, 300),
        'efficientnet-b4': (efn.EfficientNetB4, 380),
        'efficientnet-b5': (efn.EfficientNetB5, 456),
        'efficientnet-b6': (efn.EfficientNetB6, 528),
        'efficientnet-b7': (efn.EfficientNetB7, 600),
    }

    global_params = GlobalParams({
        'data_format': 'channels_last',
        'batch_norm_momentum': 0.99,
        'batch_norm_epsilon': 1e-8,
        'relu_fn': tf.nn.swish
    })

    base_model, image_size = model_map[model_name]
    base_model = base_model(include_top=False)
    base_model.trainable = False

    #layers = ('top_activation', 'block6a_expand_activation', 'block4a_expand_activation', 'block3a_expand_activation', 'block2a_expand_activation')
    layers = ('top_activation', 'block6a_expand_activation', 'block4a_expand_activation', 'block3a_expand_activation')
    layers = [base_model.get_layer(name).output for name in layers]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    features = down_stack(inputs)

    feature_shapes = []
    for l in features:
        logger.info('base model: {}, feature layer: {}, shape: {}'.format(model_name, l.name, l.shape))
        feature_shapes.append(l.shape)

    return down_stack, image_size, list(reversed(feature_shapes))

def create_model(down_stack, image_size, num_classes, num_anchors):
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    features = down_stack(inputs)

    ssd_stack = []
    for ft in features:
        ssd_head = SSD_Head(global_params, num_anchors, num_classes)
        ssd_stack.append(ssd_head(ft))

    output = tf.concat(ssd_stack, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return base_model, model, image_size
