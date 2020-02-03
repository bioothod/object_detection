import collections
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'relu_fn', 'l2_reg_weight', 'channel_axis', 'spatial_dims'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

class DarknetConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super(DarknetConv, self).__init__(**kwargs)
        self.params = params
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.relu_fn = params.relu_fn

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
                self.num_features,
                input_shape=input_shape,
                kernel_size=self.kernel_size,
                strides=self.strides,
                data_format=self.params.data_format,
                use_bias=False,
                padding=self.padding,
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.params.channel_axis,
            momentum=self.params.batch_norm_momentum,
            epsilon=self.params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)
        x = self.conv(x)
        x = self.relu_fn(x)
        return x

class DarknetConvPool(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super(DarknetConvPool, self).__init__(**kwargs)
        self.params = params
        self.num_features = num_features

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), input_shape=input_shape)
        self.conv = DarknetConv(self.params,
                self.num_features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='VALID')
        self.dropout = tf.keras.layers.Dropout(rate=self.params.spatial_dropout)

    def call(self, inputs, training):
        x = self.pad(inputs)
        x = self.conv(x, training)
        x = self.dropout(x, training)
        return x

class DarknetResidual(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super(DarknetResidual, self).__init__(**kwargs)
        self.filters = filters
        self.params = params

    def build(self, input_shape):
        self.conv0 = DarknetConv(self.params, self.filters[0], kernel_size=(1, 1), strides=(1, 1), padding='SAME', input_shape=input_shape)
        self.conv1 = DarknetConv(self.params, self.filters[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    def call(self, inputs, training):
        x = self.conv0(inputs, training)
        x = self.conv1(x, training)

        x += inputs
        return x

class DarknetRawFeatureUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, features, want_upsampling=True, **kwargs):
        super().__init__(**kwargs)

        kernel_size = (1, 1)
        self.conv_blocks = []
        for num_features in features:
            self.conv_blocks.append(DarknetConv(params, num_features, kernel_size=kernel_size, strides=(1, 1), padding='SAME'))

            if kernel_size == (1 ,1):
                kernel_size = (3, 3)
            else:
                kernel_size = (1, 1)

        self.upsample = False
        if want_upsampling:
            self.upsample = DarknetUpsampling(params, features[-1])

    def call(self, x, training):
        for conv in self.conv_blocks:
            x = conv(x, training)

        if self.upsample:
            x = self.upsample(x, training)
        return x


class DarknetBody(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(DarknetBody, self).__init__(**kwargs)
        
        # (256, 256, 3)
        self.l0a = DarknetConv(params, 32, name="l0")
        self.l0_pool = DarknetConvPool(params, 64, name="l0_pool")

        # (128, 128, 64)
        self.l1a = DarknetResidual(params, [32, 64], name="l1")
        self.l1_pool = DarknetConvPool(params, 128, name="l1_pool")

        # (64, 64, 128)
        self.l2a = DarknetResidual(params, [64, 128], name="l2a")
        self.l2b = DarknetResidual(params, [64, 128], name="l2b")
        self.l2_pool = DarknetConvPool(params, 256, name="l2_pool")

        # (32, 32, 256)
        self.l3a = DarknetResidual(params, [128, 256], name="l3a")
        self.l3b = DarknetResidual(params, [128, 256], name="l3b")
        self.l3c = DarknetResidual(params, [128, 256], name="l3c")
        self.l3d = DarknetResidual(params, [128, 256], name="l3d")
        self.l3e = DarknetResidual(params, [128, 256], name="l3e")
        self.l3f = DarknetResidual(params, [128, 256], name="l3f")
        self.l3g = DarknetResidual(params, [128, 256], name="l3g")
        self.l3h = DarknetResidual(params, [128, 256], name="l3h")
        self.l3_pool = DarknetConvPool(params, 512, name="l3_pool")
        
        # (16, 16, 512)
        self.l4a = DarknetResidual(params, [256, 512], name="l4a")
        self.l4b = DarknetResidual(params, [256, 512], name="l4b")
        self.l4c = DarknetResidual(params, [256, 512], name="l4c")
        self.l4d = DarknetResidual(params, [256, 512], name="l4d")
        self.l4e = DarknetResidual(params, [256, 512], name="l4e")
        self.l4f = DarknetResidual(params, [256, 512], name="l4f")
        self.l4g = DarknetResidual(params, [256, 512], name="l4g")
        self.l4h = DarknetResidual(params, [256, 512], name="l4h")
        self.l4_pool = DarknetConvPool(params, 1024, name="l4_pool")

        # (8, 8, 1024)
        self.l5a = DarknetResidual(params, [512, 1024], name="l5a")
        self.l5b = DarknetResidual(params, [512, 1024], name="l5b")
        self.l5c = DarknetResidual(params, [512, 1024], name="l5c")
        self.l5d = DarknetResidual(params, [512, 1024], name="l5d")
        
        self.raw0_upsample = DarknetRawFeatureUpsampling(params, [128, 256, 128], want_upsampling=False)
        self.raw1_upsample = DarknetRawFeatureUpsampling(params, [128, 256, 128])
        self.raw2_upsample = DarknetRawFeatureUpsampling(params, [256, 512, 256])
        self.raw3_upsample = DarknetRawFeatureUpsampling(params, [512, 1024, 512])

    def call(self, inputs, training):
        raw = []

        x = self.l0a(inputs, training)
        x = self.l0_pool(x, training)

        x = self.l1a(x, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2b(x, training)
        raw.append(x)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3c(x, training)
        x = self.l3d(x, training)
        x = self.l3e(x, training)
        x = self.l3f(x, training)
        x = self.l3g(x, training)
        x = self.l3h(x, training)
        raw.append(x)
        output_l3 = x
        x = self.l3_pool(x, training)

        x = self.l4a(x, training)
        x = self.l4b(x, training)
        x = self.l4c(x, training)
        x = self.l4d(x, training)
        x = self.l4e(x, training)
        x = self.l4f(x, training)
        x = self.l4g(x, training)
        x = self.l4h(x, training)
        raw.append(x)
        output_l4 = x
        x = self.l4_pool(x, training)

        x = self.l5a(x, training)
        x = self.l5b(x, training)
        x = self.l5c(x, training)
        x = self.l5d(x, training)
        raw.append(x)
        output_l5 = x

        outputs = [output_l3, output_l4, output_l5]

        x = self.raw3_upsample(raw[3], training=training)
        x = tf.concat([raw[2], x], -1)

        #x = raw[2]

        x = self.raw2_upsample(x, training=training)
        x = tf.concat([raw[1], x], -1)

        x = self.raw1_upsample(x, training=training)
        x = tf.concat([raw[0], x], -1)

        x = self.raw0_upsample(x, training=training)

        return outputs, x

class DarknetConv5(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super(DarknetConv5, self).__init__(**kwargs)

        self.conv0 = DarknetConv(params, filters[0], kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv1 = DarknetConv(params, filters[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv2 = DarknetConv(params, filters[2], kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv3 = DarknetConv(params, filters[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv4 = DarknetConv(params, filters[4], kernel_size=(1, 1), strides=(1, 1), padding='SAME')

    def call(self, input_tensor, training):
        x = self.conv0(input_tensor, training)
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        return x

class DarknetConv2(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super(DarknetConv2, self).__init__(**kwargs)

        self.conv0 = DarknetConv(params, filters[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv1 = tf.keras.layers.Conv2D(
                filters[1],
                kernel_size=(1, 1),
                strides=(1, 1),
                data_format=params.data_format,
                use_bias=True,
                padding='SAME',
                kernel_initializer='glorot_uniform')

    def call(self, input_tensor, training):
        x = self.conv0(input_tensor, training)
        x = self.conv1(x)
        return x

class DarknetUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super(DarknetUpsampling, self).__init__(**kwargs)

        self.conv = DarknetConv(params, num_features, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.upsampling = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

    def call(self, inputs, training):
        x = self.conv(inputs, training)
        x = self.upsampling(x)
        return x

class DarknetHead(tf.keras.layers.Layer):
    def __init__(self, params, num_classes, **kwargs):
        super(DarknetHead, self).__init__(**kwargs)
        num_features = 3 * (num_classes+1+4)
        
        self.stage5_conv5 = DarknetConv5(params, [512, 1024, 512, 1024, 512])
        self.stage5_conv2 = DarknetConv2(params, [1024, num_features], name="detection_layer_1")
        self.stage5_upsampling = DarknetUpsampling(params, 256)

        self.stage4_conv5 = DarknetConv5(params, [256, 512, 256, 512, 256])
        self.stage4_conv2 = DarknetConv2(params, [512, num_features], name="detection_layer_2")
        self.stage4_upsampling = DarknetUpsampling(params, 128)

        self.stage3_conv5 = DarknetConv5(params, [128, 256, 128, 256, 128])
        self.stage3_conv2 = DarknetConv2(params, [256, num_features], name="detection_layer_3")

    def call(self, stage3_in, stage4_in, stage5_in, training):
        x = self.stage5_conv5(stage5_in, training)
        stage5_output = self.stage5_conv2(x, training)

        x = self.stage5_upsampling(x, training)
        x = tf.keras.layers.concatenate([x, stage4_in])
        x = self.stage4_conv5(x, training)
        stage4_output = self.stage4_conv2(x, training)

        x = self.stage4_upsampling(x, training)
        x = tf.keras.layers.concatenate([x, stage3_in])
        x = self.stage3_conv5(x, training)
        stage3_output = self.stage3_conv2(x, training)

        return stage5_output, stage4_output, stage3_output
