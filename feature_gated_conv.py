import tensorflow as tf

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
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                bias_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
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
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                bias_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                kernel_initializer='glorot_uniform')
        self.conv1 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                kernel_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
                bias_regularizer=tf.keras.regularizers.l2(params.l2_reg_weight),
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

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(FeatureExtractor, self).__init__(self, **kwargs)

        self.c0 = TextConv(params, 32, kernel_size=(3, 3), strides=(2, 2))
        self.g0 = GatedTextConv(params, 32, kernel_size=(3, 3), strides=1)

        self.c1 = TextConv(params, 40, kernel_size=(3, 3), strides=(1, 1))
        self.g1 = GatedTextConv(params, 40, kernel_size=3, strides=1)

        self.c2 = TextConv(params, 48, kernel_size=(2, 4), strides=(2, 2))
        self.g2 = GatedTextConv(params, 48, kernel_size=3, strides=1)

        self.c3 = TextConv(params, 56, kernel_size=(3, 3), strides=(1, 1))
        self.g3 = GatedTextConv(params, 56, kernel_size=3, strides=1)

        self.c4 = TextConv(params, 64, kernel_size=(2, 4), strides=(2, 2))
        self.g4 = GatedTextConv(params, 64, kernel_size=3, strides=1)

        self.c5 = TextConv(params, 72, kernel_size=(3, 3), strides=(2, 2))
        self.g5 = GatedTextConv(params, 72, kernel_size=3, strides=1)

        self.c6 = TextConv(params, 88, kernel_size=(3, 3), strides=(1, 1))

        self.max_pooling = tf.keras.layers.MaxPooling2D((2, 2))

    def call(self, x, training):
        x = self.c0(x, training=training)
        x = self.g0(x, training=training)

        x = self.c1(x, training=training)
        x = self.g1(x, training=training)

        x = self.c2(x, training=training)
        x = self.g2(x, training=training)

        x = self.c3(x, training=training)
        x = self.g3(x, training=training)

        x = self.c4(x, training=training)
        x = self.g4(x, training=training)
        out0 = x

        x = self.c5(x, training=training)
        x = self.g5(x, training=training)
        out1 = x

        x = self.c6(x, training=training)
        x = self.max_pooling(x, training=training)
        out2 = x

        return [out0, out1, out2]
