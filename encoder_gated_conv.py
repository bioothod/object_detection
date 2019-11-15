import collections
import logging

import tensorflow as tf

import efficientnet as efn

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout'
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
def local_swish(x):
    return x * tf.nn.sigmoid(x)

class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_features, return_sequence=False, cell='lstm', **kwargs):
        super(RNNLayer, self).__init__(**kwargs)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

        cell = cell.lower()
        if cell == 'lstm':
            self.rnn = tf.keras.layers.LSTM(num_features, return_sequences=return_sequence, dropout=params.lstm_dropout)
        elif cell == 'gru':
            self.rnn = tf.keras.layers.GRU(num_features, return_sequences=return_sequence, dropout=params.lstm_dropout)
        else:
            raise NameError('unsupported rnn cell type "{}"'.format(cell))

        self.bidirectional = tf.keras.layers.Bidirectional(self.rnn, merge_mode='concat')

    def call(self, x, training):
        x = self.bn(x, training)
        x = self.bidirectional(x)

        return x

class TextConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super(TextConv, self).__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.conv = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
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
        super(GatedTextConv, self).__init__(**kwargs)
        self.conv0 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
                data_format=params.data_format,
                padding=padding,
                kernel_initializer='glorot_uniform')
        self.conv1 = tf.keras.layers.Conv2D(
                num_features,
                kernel_size=kernel_size,
                strides=strides,
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
        x0 = tf.nn.tanh(x0)

        x1 = self.conv1(x)
        x1 = tf.nn.sigmoid(x1)

        x = x0 * x1
        return x


default_char_dictionary="!\"#&\'\\()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class TextModel(tf.keras.layers.Layer):
    def __init__(self, params, max_sentence_len, dictionary=default_char_dictionary, **kwargs):
        super(TextModel, self).__init__(**kwargs)

        self.max_sentence_len = max_sentence_len

        self.c0 = TextConv(params, 16, kernel_size=(3, 3), strides=(2, 2))
        self.g0 = GatedTextConv(params, 16, kernel_size=(3, 3), strides=1)

        self.c1 = TextConv(params, 24, kernel_size=(3, 3), strides=(1, 1))
        self.g1 = GatedTextConv(params, 24, kernel_size=3, strides=1)

        self.c2 = TextConv(params, 32, kernel_size=(2, 4), strides=(2, 4))
        self.g2 = GatedTextConv(params, 32, kernel_size=3, strides=1)
        self.dropout2 = tf.keras.layers.Dropout(params.spatial_dropout)

        self.c3 = TextConv(params, 40, kernel_size=(3, 3), strides=(1, 1))
        self.g3 = GatedTextConv(params, 40, kernel_size=3, strides=1)
        self.dropout3 = tf.keras.layers.Dropout(params.spatial_dropout)

        self.c4 = TextConv(params, 48, kernel_size=(2, 4), strides=(2, 4))
        self.g4 = GatedTextConv(params, 48, kernel_size=3, strides=1)
        self.dropout4 = tf.keras.layers.Dropout(params.spatial_dropout)
        self.upsampling4 = tf.keras.layers.UpSampling2D((2, 4))

        self.c5 = TextConv(params, 64, kernel_size=(3, 3), strides=(1, 1))

        self.max_pooling = tf.keras.layers.MaxPooling2D((1, 2))

        self.ct = tf.keras.layers.Conv2D(
                max_sentence_len,
                kernel_size=1,
                strides=1,
                data_format=params.data_format,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_uniform')


        self.lstm0 = RNNLayer(params, 128, return_sequence=True, cell='lstm')
        self.dense0 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128))

        self.lstm1 = RNNLayer(params, 128, return_sequence=True, cell='gru')
        self.dense1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(dictionary) + 1))

    def call(self, x, training):
        x = self.c0(x, training=training)
        x = self.g0(x, training=training)

        x = self.c1(x, training=training)
        x = self.g1(x, training=training)

        x = self.c2(x, training=training)
        x = self.g2(x, training=training)
        x = self.dropout2(x, training)

        x = self.c3(x, training=training)
        g3x = self.g3(x, training=training)
        x = self.dropout3(g3x, training)

        x = self.c4(x, training=training)
        x = self.g4(x, training=training)
        up4x = self.upsampling4(x, training=training)

        x = tf.concat([g3x, up4x], -1)

        x = self.dropout4(x, training)

        x = self.c5(x, training=training)
        x = self.max_pooling(x, training=training)
        logger.info('max_pooling: {}'.format(x.shape))

        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], shape[1] * shape[2], shape[3]])
        logger.info('reshape: {}'.format(x.shape))

        x = tf.transpose(x, [0, 2, 1])
        x = tf.expand_dims(x, 2)
        x = self.ct(x)
        x = tf.squeeze(x, 2)
        x = tf.transpose(x, [0, 2, 1])
        logger.info('after ct: {}'.format(x.shape))

        x = self.lstm0(x, training=training)
        x = self.dense0(x, training=training)

        x = self.lstm1(x, training=training)
        x = self.dense1(x, training=training)

        logger.info('lstm: {}'.format(x.shape))

        return x

def create_params(model_name):
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
        'batch_norm_epsilon': 1e-8,
        'channel_axis': channel_axis,
        'spatial_dims': spatial_dims,
        'model_name': model_name,
        'obj_score_threshold': 0.3,
        'lstm_dropout': 0.3,
        'spatial_dropout': 0.3,
    }

    params = GlobalParams(**params)

    return params


def create_text_recognition_model(model_name, max_sentence_len):
    params = create_params(model_name)
    model = TextModel(params, max_sentence_len)
    return model
