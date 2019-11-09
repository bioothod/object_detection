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

class EfnBody(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(EfnBody, self).__init__(**kwargs)

        self.image_size = efn.efficientnet_params(params.model_name)[2]

        efn_param_keys = efn.GlobalParams._fields
        efn_params = {}
        for k, v in params._asdict().items():
            if k in efn_param_keys:
                efn_params[k] = v

        self.base_model = efn.build_model(model_name=params.model_name, override_params=efn_params)

        self.reduction_indexes = [3, 4, 5]

    def call(self, inputs, training=True):
        self.endpoints = []

        outputs = self.base_model(inputs, training=training, features_only=True)


        for reduction_idx in self.reduction_indexes:
            endpoint = self.base_model.endpoints['reduction_{}'.format(reduction_idx)]
            self.endpoints.append(endpoint)

        return self.endpoints

class DarknetConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', want_max_pool=False, **kwargs):
        super(DarknetConv, self).__init__(**kwargs)
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

        self.num_features = num_features
        self.conv = DarknetConv(params, num_features, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.upsampling = tf.keras.layers.UpSampling2D(2)

    def call(self, inputs, training):
        x = self.conv(inputs, training)
        x = self.upsampling(x)
        return x

class EfnHead(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(EfnHead, self).__init__(**kwargs)

        num_features = 3 * (1 + 4)

        self.stage2_conv5 = DarknetConv5(params, [512, 1024, 512, 1024, 512])
        self.stage2_conv2 = DarknetConv2(params, [1024, num_features], name="detection_layer_1")
        self.stage2_upsampling = DarknetUpsampling(params, 256)

        self.stage1_conv5 = DarknetConv5(params, [256, 512, 256, 512, 256])
        self.stage1_conv2 = DarknetConv2(params, [512, num_features], name="detection_layer_2")
        self.stage1_upsampling = DarknetUpsampling(params, 128)

        self.stage0_conv5 = DarknetConv5(params, [128, 256, 128, 256, 128])
        self.stage0_conv2 = DarknetConv2(params, [256, num_features], name="detection_layer_3")

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

    def call(self, stage0_in, stage1_in, stage2_in, training=True):
        x = self.stage2_conv5(stage2_in, training)
        stage2_output = self.stage2_conv2(x, training)

        x = self.stage2_upsampling(x, training)

        diff = x.shape[1] - stage1_in.shape[1]
        if diff > 0:
            stage1_in = self.pads[diff](stage1_in)
        if diff < 0:
            x = self.pads[-diff](x)

        x = tf.keras.layers.concatenate([x, stage1_in])
        x = self.stage1_conv5(x, training)
        stage1_output = self.stage1_conv2(x, training)

        x = self.stage1_upsampling(x, training)

        diff = x.shape[1] - stage0_in.shape[1]
        if diff > 0:
            stage0_in = self.pads[diff](stage0_in)
        if diff < 0:
            x = self.pads[-diff](x)

        x = tf.keras.layers.concatenate([x, stage0_in])
        x = self.stage0_conv5(x, training)
        stage0_output = self.stage0_conv2(x, training)

        return stage2_output, stage1_output, stage0_output

class EfnYolo(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(EfnYolo, self).__init__(**kwargs)

        self.body = EfnBody(params)
        self.head = EfnHead(params)

        self.image_size = self.body.image_size
        os = {
                'efficientnet-b0': [7, 14, 28],
                'efficientnet-b1': [8, 16, 32],
                'efficientnet-b2': [9, 18, 36],
                'efficientnet-b4': [12, 24, 48],
                'efficientnet-b6': [17, 34, 68],
        }
        self.output_sizes = os.get(params.model_name, None)

    def call(self, inputs, training):
        l = self.body(inputs, training)
        f2, f1, f0 = self.head(l[0], l[1], l[2], training)

        self.output_sizes = [tf.shape(f2)[1], tf.shape(f1)[1], tf.shape(f0)[1]]

        batch_size = tf.shape(inputs)[0]
        outputs = []
        for output in [f2, f1, f0]:
            flat = tf.reshape(output, [batch_size, -1, 4+1])
            outputs.append(flat)

        outputs = tf.concat(outputs, axis=1)
        return outputs

class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_features, return_sequence=False, **kwargs):
        super(LSTMLayer, self).__init__(**kwargs)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

        self.rnn = tf.keras.layers.LSTM(num_features, return_sequences=return_sequence, dropout=params.lstm_dropout)
        self.bidirectional = tf.keras.layers.Bidirectional(self.rnn, merge_mode='concat')

    def call(self, x, training):
        x = self.bn(x, training)
        x = self.bidirectional(x)

        return x

class TextConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', pool_size=(1, 1), pool_strides=(1, 1), **kwargs):
        super(TextConv, self).__init__(**kwargs)
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

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, data_format=params.data_format, padding='VALID')

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.relu_fn(x)

        x = self.max_pool(x)

        return x


default_char_dictionary="!\"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class TextModel(tf.keras.layers.Layer):
    def __init__(self, params, max_sentence_len, dictionary=default_char_dictionary, **kwargs):
        super(TextModel, self).__init__(**kwargs)

        self.max_sentence_len = max_sentence_len

        self.c0 = TextConv(params, 64, kernel_size=5, strides=1, pool_size=(1,1), pool_strides=(1, 1))
        self.c1 = TextConv(params, 128, kernel_size=5, strides=1, pool_size=(2,2), pool_strides=(2, 2))
        self.c2 = TextConv(params, 256, kernel_size=3, strides=1, pool_size=(1,2), pool_strides=(1, 2))
        self.c3 = TextConv(params, 256, kernel_size=3, strides=1, pool_size=(1,2), pool_strides=(1, 2))
        self.c4 = TextConv(params, 512, kernel_size=3, strides=1, pool_size=(1,2), pool_strides=(1, 2))

        self.dropout = tf.keras.layers.SpatialDropout2D(params.spatial_dropout)

        self.lstm0 = LSTMLayer(params, 256, return_sequence=True)
        self.lstm1 = LSTMLayer(params, 256, return_sequence=True)

        self.out_conv = tf.keras.layers.Conv2D(len(dictionary) + 1, kernel_size=3, strides=1, padding='SAME')


    def call(self, inputs, training):
        x = self.c0(inputs, training=training)
        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        x = self.c3(x, training=training)
        x = self.c4(x, training=training)

        outputs = self.dropout(x, training)
        #outputs = x

        shape = tf.shape(outputs)
        x = tf.reshape(outputs, [shape[0], shape[1] * shape[2], shape[3]])
        logger.info('inputs: {}, outputs: {}, x: {}'.format(inputs.shape, outputs.shape, x.shape))

        x = self.lstm0(x, training)
        x = self.lstm1(x, training)
        logger.info('lstm: {}'.format(x.shape))

        # BxTx2H -> BxTx1x2H
        x = tf.expand_dims(x, 2)
        x = self.out_conv(x)
        x = tf.squeeze(x, 2)

        #x = tf.keras.activations.softmax(x, -1)

        return x

    def decode(self, inputs):
        logits = self.call(inputs, training=False)

        # BxTxC -> TxBxC
        logits = tf.transpose(logits, [1, 0, 2])

        #decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=self.max_sentence_len, beam_width=50)

        decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=self.max_sentence_len)

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
        'batch_norm_epsilon': 1e-3,
        'channel_axis': channel_axis,
        'spatial_dims': spatial_dims,
        'model_name': model_name,
        'obj_score_threshold': 0.3,
        'lstm_dropout': 0.3,
        'spatial_dropout': 0.3,
    }

    params = GlobalParams(**params)

    return params


def create_model(model_name):
    params = create_params(model_name)
    model = EfnYolo(params)
    return model

def create_text_recognition_model(model_name, max_sentence_len):
    params = create_params(model_name)
    model = TextModel(params, max_sentence_len)
    return model
