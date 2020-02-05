import logging

import tensorflow as tf

logger = logging.getLogger('detection')

class TextConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super(TextConv, self).__init__(self, **kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.conv = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
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

class GatedTextConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super(GatedTextConv, self).__init__(self, **kwargs)
        self.conv0 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                kernel_initializer='glorot_uniform')
        self.conv1 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                data_format=params.data_format,
                padding=padding,
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)

        x0 = self.conv0(x)
        x0 = tf.nn.tanh(x0)

        x1 = self.conv1(x)
        x1 = tf.nn.sigmoid(x1)

        x = x0 * x1
        return x

class GatedBlock(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super().__init__(**kwargs)

        self.conv = TextConv(params, num_features, kernel_size=kernel_size, strides=strides, padding=padding)
        self.gate = GatedTextConv(params, num_features, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, inputs, training):
        x = self.conv(inputs, training=training)
        x = self.gate(x)
        return x

class GatedBlockResidual(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.conv0 = GatedBlock(params, num_features[0], kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv1 = GatedBlock(params, num_features[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    def call(self, inputs, training):
        x = self.conv0(inputs, training)
        x = self.conv1(x, training)

        x += inputs
        return x

class GatedBlockPool(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))

        self.conv = TextConv(params,
                        num_features,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='VALID')

        self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)

    def call(self, inputs, training):
        x = self.pad(inputs)
        x = self.conv(x, training)
        x = self.dropout(x, training)
        return x

class GatedBlockUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.conv = GatedBlock(params, num_features, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.upsampling = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

    def call(self, inputs, training):
        x = self.conv(inputs, training)
        x = self.upsampling(x)
        return x

class GatedBlockConvUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, features, want_upsampling=True, **kwargs):
        super().__init__(**kwargs)

        kernel_size = (1, 1)
        self.conv_blocks = []
        for num_features in features:
            self.conv_blocks.append(GatedBlock(params, num_features, kernel_size=kernel_size, strides=(1, 1), padding='SAME'))

            if kernel_size == (1 ,1):
                kernel_size = (3, 3)
            else:
                kernel_size = (1, 1)

        self.upsample = False
        if want_upsampling:
            self.upsample = GatedBlockUpsampling(params, features[-1])

    def call(self, x, training):
        for conv in self.conv_blocks:
            x = conv(x, training)

        if self.upsample:
            x = self.upsample(x, training)
        return x

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(FeatureExtractor, self).__init__(self, **kwargs)

        self.blocks = []

        self.blocks.append(GatedBlock(params, 32))
        self.blocks.append(GatedBlock(params, 64))
        self.blocks.append(GatedBlockPool(params, 64))

        self.blocks.append(GatedBlockResidual(params, [32, 64]))
        self.blocks.append(GatedBlockResidual(params, [32, 64]))
        self.blocks.append(GatedBlockPool(params, 128))

        self.blocks.append(GatedBlockResidual(params, [64, 128]))
        self.blocks.append(GatedBlockResidual(params, [64, 128]))
        self.blocks.append(GatedBlockResidual(params, [64, 128]))
        self.blocks.append(GatedBlockResidual(params, [64, 128], name='raw0'))
        self.blocks.append(GatedBlockPool(params, 256, name='output0'))

        self.blocks.append(GatedBlockResidual(params, [128, 256]))
        self.blocks.append(GatedBlockResidual(params, [128, 256]))
        self.blocks.append(GatedBlockResidual(params, [128, 256]))
        self.blocks.append(GatedBlockResidual(params, [128, 256]))
        self.blocks.append(GatedBlockResidual(params, [128, 256]))
        self.blocks.append(GatedBlockResidual(params, [128, 256], name='raw1'))
        self.blocks.append(GatedBlockPool(params, 512, name='output1'))

        self.blocks.append(GatedBlockResidual(params, [256, 512]))
        self.blocks.append(GatedBlockResidual(params, [256, 512], name='raw2'))
        self.blocks.append(GatedBlockPool(params, 1024, name='output2'))

        self.raw0_upsample = GatedBlockConvUpsampling(params, [128] , want_upsampling=False)
        self.raw1_upsample = GatedBlockConvUpsampling(params, [256])
        self.raw2_upsample = GatedBlockConvUpsampling(params, [512])
        #self.raw3_upsample = GatedBlockConvUpsampling(params, [256, 512])

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

        x = self.raw2_upsample(x, training=training)
        x = tf.concat([raw[1], x], -1)

        x = self.raw1_upsample(x, training=training)
        x = tf.concat([raw[0], x], -1)

        x = self.raw0_upsample(x, training=training)

        return outputs, x
