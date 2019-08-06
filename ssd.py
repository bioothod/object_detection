from efficientnet import conv_kernel_initializer

class StdConv(tf.keras.layers.Layer):
    def __init__(self, global_params, num_filters, strides=2, dropout_rate=0.1):
        super(StdConv, self).__init__()

        self._num_filters = num_filters
        self._strides = strides
        self._dropout_rate = dropout_rate

        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._data_format = global_params.data_format
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._build()

    def _build(self):
        self._c0 = tf.keras.layers.Conv2D(
            filters=self._num_filters,
            kernel_size=3,
            strides=self._strides,
            data_format=self._data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self._bn0 = batchnorm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        self._dropout0 = tf.keras.layers.Dropout(self._dropout_rate)

    def call(self, inputs, training=True):
        return self._dropout0(self._bn0(self._c0(inputs), training=training))

def flaten_conv(global_params, x, k):
    if global_params.data_format == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])

    return tf.reshape(x, (-1, tf.shape(x)[-1] // k))

class OutConv(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes):
        super(OutConv, self).__init__()

        self._k = k
        self._num_classes = num_classes

        self._data_format = global_params.data_format

        self._build()

    def _build(self):
        self._c1 = tf.keras.layers.Conv2D(
            filters=self._num_classes*self._k,
            kernel_size=3,
            strides=1,
            data_format=self._data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        self._c2 = tf.keras.layers.Conv2D(
            filters=4*self._k,
            kernel_size=3,
            strides=1,
            data_format=self._data_format,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

    def call(self, inputs, training=True):
        return [flatten_conv(global_params, self._c1(inputs), self.k), flatten_conv(global_params, self._c2(inputs), self.k)]

class SSD_Head(tf.keras.models.Model):
    def __init__(self, global_params, k, num_classes):
        super(SSD_Head, self).__init__()

        self._k = k
        self._num_classes = num_classes

        self._relu_fn = global_params.relu_fn or tf.nn.swish

        self._build()

    def _build(self):
        self._dropout = tf.keras.layers.Dropout(0.25)
        self._sc0 = StdConv(global_params, 256, strides=1)
        self._sc2 = StdConv(global_params, 256)
        self._out = OutConv(global_params, self._k, self._num_classes)

    def call(self, inputs, training=True):
        x = self._relu_fn(inputs)
        x = self._dropout(x)
        x = self._sc0(x)
        x = self._sc2(x)
        x = self._out(x)

        return x
