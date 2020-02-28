import tensorflow as tf


class LettersConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=3, strides=1, padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.conv = tf.keras.layers.Conv1D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)
        x = self.conv(x)
        x = self.relu_fn(x)

        return x

class GatedLettersConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=3, strides=1, padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.conv0 = tf.keras.layers.Conv1D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                data_format=params.data_format,
                padding=padding)
        self.conv1 = tf.keras.layers.Conv1D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=False,
                data_format=params.data_format,
                padding=padding)

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

class GatedLettersResidual(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.convs = []
        kernel_size = 1
        for num in num_features:
            conv = LettersConv(params, num, kernel_size=kernel_size, strides=1, padding='SAME')
            self.convs.append(conv)

            if kernel_size == 1:
                kernel_size = 3
            else:
                kernel_size = 1

        self.gate = GatedLettersConv(params, num_features[-1], kernel_size=3, strides=1, padding='SAME')

    def call(self, inputs, training):
        x = inputs
        for conv in self.convs:
            x = conv(x, training)

        #gx = self.gate(x, training)

        outputs = inputs + x

        return outputs

class LettersPool(tf.keras.layers.Layer):
    def __init__(self, params, num_features, want_dropout=True, strides=2, **kwargs):
        super().__init__(**kwargs)

        self.conv = LettersConv(params,
                        num_features,
                        kernel_size=3,
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

class LettersDetector(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)

        self.res0 = GatedLettersResidual(params, [128, 128], name='res0')

        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training):
        x = self.dropout(inputs, training)
        x = self.res0(x, training)

        x = self.avg_pool(x)
        return x
