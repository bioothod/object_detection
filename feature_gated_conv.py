import tensorflow as tf

import darknet

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
    def __init__(self, params, num_features, kernel_size, strides, dropout_rate=0, upsampling=None, **kwargs):
        super().__init__(**kwargs)

        self.conv = TextConv(params, num_features, kernel_size=kernel_size, strides=strides)
        self.gate = GatedTextConv(params, num_features, kernel_size=3, strides=1)

        self.dropout = None
        if dropout_rate and dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(params.spatial_dropout)

        self.upsampling = None
        if upsampling and upsampling != 0 and upsampling != (0, 0):
            self.upsampling = tf.keras.layers.UpSampling2D(upsampling)

    def call(self, inputs, training):
        x = self.conv(inputs, training=training)
        x = self.gate(x)
        if self.dropout:
            x = self.dropout(x, training=training)

        if self.upsampling:
            x = self.upsampling(x)

        return x


class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(FeatureExtractor, self).__init__(self, **kwargs)

        num_features = 16

        self.blocks = []

        self.blocks.append(GatedBlock(params, num_features, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0))
        self.blocks.append(GatedBlock(params, num_features*2, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0))
        self.blocks.append(GatedBlock(params, num_features*4, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0))
        self.blocks.append(tf.keras.layers.MaxPooling2D((2, 2)))
        self.blocks.append(GatedBlock(params, num_features*8, kernel_size=(3, 3), strides=(1, 1), dropout_rate=params.spatial_dropout))
        self.blocks.append(GatedBlock(params, num_features*8, kernel_size=(3, 3), strides=(1, 1), dropout_rate=params.spatial_dropout))
        self.blocks.append(tf.keras.layers.MaxPooling2D((2, 2)))
        self.blocks.append(GatedBlock(params, num_features*16, kernel_size=(3, 3), strides=(1, 1), dropout_rate=params.spatial_dropout))
        self.blocks.append(GatedBlock(params, num_features*16, kernel_size=(1, 1), strides=(1, 1), dropout_rate=params.spatial_dropout, name='raw0'))
        self.blocks.append(tf.keras.layers.MaxPooling2D((2, 2), name='output2'))
        self.blocks.append(GatedBlock(params, num_features*32, kernel_size=(3, 3), strides=(1, 1), dropout_rate=params.spatial_dropout))
        self.blocks.append(GatedBlock(params, num_features*32, kernel_size=(1, 1), strides=(1, 1), dropout_rate=params.spatial_dropout, name='raw1'))
        self.blocks.append(tf.keras.layers.MaxPooling2D((2, 2), name='output4'))
        self.blocks.append(GatedBlock(params, num_features*32, kernel_size=(3, 3), strides=(1, 1), dropout_rate=params.spatial_dropout))
        self.blocks.append(GatedBlock(params, num_features*32, kernel_size=(1, 1), strides=(1, 1), dropout_rate=params.spatial_dropout, name='raw2'))
        self.blocks.append(tf.keras.layers.MaxPooling2D((2, 2), name='output4'))

        self.raw1_upsample = darknet.DarknetUpsampling(params, 256)
        self.raw2_upsample = darknet.DarknetUpsampling(params, 256)
        #self.raw3_upsample = darknet.DarknetUpsampling(params, 512)

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

        return outputs, x
