import logging

import numpy as np
import tensorflow as tf

from efficientnet import conv_kernel_initializer

logger = logging.getLogger('objdet')
logger.setLevel(logging.INFO)

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

class SSD(tf.keras.models.Model):
    def __init__(self, global_params, endpoints, k, num_classes):
        super(SSD, self).__init__()

        self.global_params = global_params
        self.k = k
        self.num_classes = num_classes

        self.endpoints = None
        self.input_endpoints = endpoints

        self._build()

    def _build(self):
        self.endpoints = []

        for name, endpoint in self.input_endpoints:
            with tf.name_scope('ssd_{}'.format(name)):
                head = SSD_Head(self.global_params, self.k, self.num_classes)

                self.endpoints.append((name, endpoint, head))

    def call(self, base_model, inputs, training=True):
        if self.global_params.data_format == 'channels_first':
            inputs = tf.transpose(inputs, (0, 3, 1, 2))

        x = base_model(inputs, training)

        outputs = []
        for name, endpoint, ssd_head in self.endpoints:
            so = ssd_head(endpoint)
            outputs.append(so)

        return outputs

class SSD_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name=None):
        super(SSD_Loss, self).__init__(reduction, name)

    def __call__(y_true, y_pred, sample_weight=None):
        logger.info('true: {}, pred: {}'.format(y_true.shape, y_pred.shape))
