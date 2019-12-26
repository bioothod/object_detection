import collections
import logging

import tensorflow as tf

import darknet
import encoder_efficientnet as efn_encoder
import feature_gated_conv as features

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout',
    'dictionary_size'
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
def local_swish(x):
    return x * tf.nn.sigmoid(x)

class EncoderConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', want_max_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.relu_fn = params.relu_fn or local_swish

        self.conv = tf.keras.layers.Conv2D(
                self.num_features,
                kernel_size=self.kernel_size,
                strides=self.strides,
                data_format=params.data_format,
                use_bias=False,
                padding=self.padding,
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

        self.max_pool = None
        if want_max_pool:
            self.max_pool = tf.keras.layers.MaxPool2D((2, 2), data_format=params.data_format)

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.relu_fn(x)

        if self.max_pool is not None:
            x = self.max_pool(x)

        return x

class EncoderConv5(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super().__init__(**kwargs)

        self.conv0 = EncoderConv(params, filters[0], kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv1 = EncoderConv(params, filters[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv2 = EncoderConv(params, filters[2], kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv3 = EncoderConv(params, filters[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv4 = EncoderConv(params, filters[4], kernel_size=(1, 1), strides=(1, 1), padding='SAME')

    def call(self, input_tensor, training):
        x = self.conv0(input_tensor, training)
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        return x

class HeadLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_features, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.conv0 = EncoderConv(params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

        self.outputs = []
        for n in num_outputs:
            n *= 3

            out = tf.keras.layers.Conv2D(
                n,
                kernel_size=(1, 1),
                strides=(1, 1),
                data_format=params.data_format,
                use_bias=True,
                padding='SAME',
                kernel_initializer='glorot_uniform')

            self.outputs.append(out)

    def call(self, input_tensor, training):
        x = self.conv0(input_tensor, training=training)

        outputs = []
        for f in self.outputs:
            out = f(x)
            outputs.append(out)

        return outputs

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, params, input_filters, **kwargs):
        super().__init__(**kwargs)

        char_features = [1, 2*4, params.dictionary_size]
        word_features = [1, 2*4]

        self.conv0 = EncoderConv(params, input_filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

        self.char_output = HeadLayer(params, input_filters, char_features)
        self.word_output = HeadLayer(params, input_filters, word_features)

    def call(self, input_tensor, training):
        x = self.conv0(input_tensor, training)

        char_out = self.char_output(x, training=training)
        word_out = self.word_output(x, training=training)

        outputs = tf.concat(char_out + word_out, -1)
        return outputs

class DarknetUpsampling(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.conv = EncoderConv(params, num_features, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.upsampling = tf.keras.layers.UpSampling2D(2)

    def call(self, inputs, training):
        x = self.conv(inputs, training)
        x = self.upsampling(x)
        return x

class Head(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.s2_input = EncoderConv5(params, [512, 1024, 512, 1024, 512])
        self.s2_output = OutputLayer(params, 512, name="detection_layer_2")
        self.up2 = DarknetUpsampling(params, 512)

        self.s1_input = EncoderConv5(params, [256, 512, 256, 512, 256])
        self.s1_output = OutputLayer(params, 256, name="detection_layer_1")
        self.up1 = DarknetUpsampling(params, 256)

        self.s0_input = EncoderConv5(params, [128, 256, 128, 256, 128])
        self.s0_output = OutputLayer(params, 128, name="detection_layer_0")

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

    def call(self, in0, in1, in2, training=True):
        x2 = in2
        x2 = self.s2_input(x2)
        up2 = self.up2(x2)
        out2 = self.s2_output(x2, training=training)

        diff = up2.shape[1] - in1.shape[1]
        if diff > 0:
            in1 = self.pads[diff](in1)
        if diff < 0:
            up2 = self.pads[-diff](up2)

        x1 = tf.keras.layers.concatenate([up2, in1])
        x1 = self.s1_input(x1)
        up1 = self.up1(x1)
        out1 = self.s1_output(x1, training=training)

        diff = up1.shape[1] - in0.shape[1]
        if diff > 0:
            in0 = self.pads[diff](in0)
        if diff < 0:
            up1 = self.pads[-diff](up1)

        x0 = tf.keras.layers.concatenate([up1, in0])
        x0 = self.s0_input(x0)
        out0 = self.s0_output(x0, training=training)

        return out2, out1, out0

class Encoder(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = 1 + 2*4 + params.dictionary_size + 1 + 2*4

        if params.model_name.startswith('darknet'):
            self.body = darknet.DarknetBody(params)
        elif params.model_name.startswith('gated_conv'):
            self.body = features.FeatureExtractor(params)
        elif params.model_name.startswith('efficientnet-'):
            self.body = efn_encoder.EfnBody(params)
        else:
            raise NameError('unsupported model name {}'.format(params.model_name))

        self.head = Head(params)

        self.output_sizes = None

    def call(self, inputs, training):
        l = self.body(inputs, training)
        f2, f1, f0 = self.head(l[0], l[1], l[2], training)

        self.output_sizes = [tf.shape(f2)[1], tf.shape(f1)[1], tf.shape(f0)[1]]

        batch_size = tf.shape(inputs)[0]
        outputs = []
        for output in [f2, f1, f0]:
            flat = tf.reshape(output, [batch_size, -1, self.num_classes])
            outputs.append(flat)

        outputs = tf.concat(outputs, axis=1)
        return outputs

def create_params(model_name, dictionary_size):
    data_format='channels_last'

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    params = {
        'data_format': data_format,
        'relu_fn': local_swish,
        'batch_norm_momentum': 0.99,
        'batch_norm_epsilon': 1e-3,
        'channel_axis': channel_axis,
        'spatial_dims': spatial_dims,
        'model_name': model_name,
        'obj_score_threshold': 0.3,
        'lstm_dropout': 0.3,
        'spatial_dropout': 0.3,
        'dictionary_size': dictionary_size,
    }

    params = GlobalParams(**params)

    return params


def create_model(model_name, dictionary_size):
    params = create_params(model_name, dictionary_size)
    model = Encoder(params)
    return model
