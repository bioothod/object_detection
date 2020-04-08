import logging

import tensorflow as tf

logger = logging.getLogger('detection')

class TextConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.conv = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                use_bias=False,
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)
        x = self.conv(x)
        x = self.relu_fn(x)

        return x

class GatedConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.conv0 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                data_format=params.data_format,
                padding=padding,
                kernel_initializer='glorot_uniform')
        self.conv1 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                data_format=params.data_format,
                padding=padding,
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)

        x0 = self.conv0(x)
        x0 = tf.nn.sigmoid(x0)

        x1 = self.conv1(x)
        x1 = tf.nn.tanh(x1)

        return x0 * x1

class GatedBlockResidual(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size1=(1, 1), kernel_size2=(3, 3), **kwargs):
        super().__init__(**kwargs)

        self.convs = []
        kernel_size = kernel_size1
        for num in num_features:
            conv = TextConv(params, num, kernel_size=kernel_size, strides=(1, 1), padding='SAME')
            self.convs.append(conv)

            if kernel_size == kernel_size1:
                kernel_size = kernel_size2
            else:
                kernel_size = kernel_size1

        self.gate = GatedConv(params, num_features[-1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    def call(self, inputs, training):
        x = inputs
        for conv in self.convs:
            x = conv(x, training)

        gx = self.gate(x, training)

        outputs = inputs + x + gx

        return outputs

class BlockPool(tf.keras.layers.Layer):
    def __init__(self, params, num_features, want_dropout=True, strides=(2, 2), **kwargs):
        super().__init__(**kwargs)

        self.conv = TextConv(params,
                        num_features,
                        kernel_size=(3, 3),
                        strides=strides,
                        padding='SAME')

        self.dropout = None
        if want_dropout:
            self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)

    def call(self, x, training):
        if self.dropout:
            x = self.dropout(x, training)

        x = self.conv(x, training)
        return x

class BlockConvUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, features, want_upsampling=True, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)

        self.conv_blocks = []
        kernel_size = (2, 2)
        for num_features in features:
            self.conv_blocks.append(TextConv(params, num_features, kernel_size=kernel_size, strides=(1, 1), padding='SAME'))

            if kernel_size == (2, 2):
                kernel_size = (3, 3)
            else:
                kernel_size = (2, 2)

        self.upsampling = False
        if want_upsampling:
            self.upsampling = tf.keras.layers.Conv2DTranspose(features[-1], kernel_size=(4, 4), strides=2, padding='same', use_bias=False)

    def call(self, inputs, training):
        inputs = self.dropout(inputs, training)
        x = inputs

        for conv in self.conv_blocks:
            x = conv(x, training)

        if self.upsampling:
            x = self.upsampling(x)

        return x

class FeatureExtractor(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.blocks = []

        self.blocks.append(TextConv(params, 16, kernel_size=(4, 4), name='l0_conv0'))
        self.blocks.append(TextConv(params, 32, kernel_size=(4, 4), name='l0_conv1'))
        self.blocks.append(BlockPool(params, 64, name='l0_pool'))

        self.blocks.append(GatedBlockResidual(params, [64, 64], kernel_size2=(4, 4), name='l1_res0'))
        self.blocks.append(BlockPool(params, 96, name='l1_pool'))

        self.blocks.append(GatedBlockResidual(params, [96, 96], kernel_size2=(4, 4), name='l2_res0_raw0'))
        self.blocks.append(BlockPool(params, 128, name='l2_pool_output0'))

        self.blocks.append(GatedBlockResidual(params, [128, 128], name='l3_res0_raw1'))
        self.blocks.append(BlockPool(params, 256, name='l3_pool_output1'))

        self.blocks.append(GatedBlockResidual(params, [256, 256], name='l4_res0_raw2'))
        self.blocks.append(BlockPool(params, 512, name='l4_pool_output2'))

        self.raw0_upsampling = BlockConvUpsampling(params, [256], want_upsampling=False)
        self.raw1_upsampling = BlockConvUpsampling(params, [256])
        self.raw2_upsampling = BlockConvUpsampling(params, [256])

    def call(self, x, training):
        outputs = []
        raw = []
        for block in self.blocks:
            x = block(x, training=training)
            if 'output' in block.name:
                outputs.append(x)
            if 'raw' in block.name:
                raw.append(x)

        #x = self.raw3_upsample(raw[3], training=training)
        #x = tf.concat([raw[2], x], -1)

        x = raw[2]

        x = self.raw2_upsampling(x, training=training)
        x = tf.concat([raw[1], x], -1)

        x = self.raw1_upsampling(x, training=training)
        x = tf.concat([raw[0], x], -1)

        x = self.raw0_upsampling(x, training=training)

        return outputs, x
