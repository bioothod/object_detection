import collections
import logging

import tensorflow as tf
import tensorflow_addons as tfa

import attention
import darknet
import encoder_efficientnet as efn_encoder
import feature_gated_conv as features
import preprocess

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout',
    'dictionary_size', 'max_word_len', 'pad_value'
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

class EncoderConvList(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super().__init__(**kwargs)

        self.convs = []
        idx = 0
        for num_filters in filters:
            kernel_size = 3
            if idx % 2 == 0:
                kernel_size = 1
            idx += 1

            conv = EncoderConv(params, num_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='SAME')
            self.convs.append(conv)

    def call(self, x, training):
        for conv in self.convs:
            x = conv(x, training)
        return x

class HeadLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

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

    def call(self, x, training):
        outputs = []
        for f in self.outputs:
            out = f(x)
            outputs.append(out)

        outputs = tf.concat(outputs, -1)
        return outputs

class Head(tf.keras.layers.Layer):
    def __init__(self, params, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.s2_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s2_input = EncoderConvList(params, [512, 1024, 512, 1024, 512])
        self.s2_output = HeadLayer(params, num_classes, name="detection_layer_2")
        self.up2 = tf.keras.layers.UpSampling2D(2)

        self.s1_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s1_input = EncoderConvList(params, [256, 512, 256, 512, 256])
        self.s1_output = HeadLayer(params, num_classes, name="detection_layer_1")
        self.up1 = tf.keras.layers.UpSampling2D(2)

        self.s0_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s0_input = EncoderConvList(params, [128, 256, 128, 256, 128])
        self.s0_output = HeadLayer(params, num_classes, name="detection_layer_0")

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

    def call(self, in0, in1, in2, training=True):
        x2 = self.s2_input(in2)
        up2 = self.up2(x2)
        x2 = self.s2_dropout(x2, training=training)
        out2 = self.s2_output(x2, training=training)

        diff = up2.shape[1] - in1.shape[1]
        if diff > 0:
            in1 = self.pads[diff](in1)
        if diff < 0:
            up2 = self.pads[-diff](up2)

        x1 = tf.keras.layers.concatenate([up2, in1])
        x1 = self.s1_input(x1)
        up1 = self.up1(x1)
        x1 = self.s1_dropout(x1, training=training)
        out1 = self.s1_output(x1, training=training)

        diff = up1.shape[1] - in0.shape[1]
        if diff > 0:
            in0 = self.pads[diff](in0)
        if diff < 0:
            up1 = self.pads[-diff](up1)

        x0 = tf.keras.layers.concatenate([up1, in0])
        x0 = self.s0_input(x0)
        x0 = self.s1_dropout(x0, training=training)
        out0 = self.s0_output(x0, training=training)

        return [out2, out1, out0]

class ConcatFeatures(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.s2_input = EncoderConv(params, 512)
        self.up2 = tf.keras.layers.UpSampling2D(2)

        self.s1_input = EncoderConv(params, 256)
        self.up1 = tf.keras.layers.UpSampling2D(2)

        self.s0_input = EncoderConv(params, 128)

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

    def call(self, in0, in1, in2, training=True):
        x2 = self.s2_input(in2)
        up2 = self.up2(x2)

        diff = up2.shape[1] - in1.shape[1]
        if diff > 0:
            in1 = self.pads[diff](in1)
        if diff < 0:
            up2 = self.pads[-diff](up2)

        x1 = tf.keras.layers.concatenate([up2, in1])
        x1 = self.s1_input(x1)
        up1 = self.up1(x1)

        diff = up1.shape[1] - in0.shape[1]
        if diff > 0:
            in0 = self.pads[diff](in0)
        if diff < 0:
            up1 = self.pads[-diff](up1)

        x0 = tf.keras.layers.concatenate([up1, in0])
        x0 = self.s0_input(x0)

        return x0

@tf.function(experimental_relax_shapes=True)
def run_crop_and_rotation(features, x, y, xmin, ymin, xmax, ymax):
    features_for_one_crop = features[ymin : ymax + 1, xmin : xmax + 1, :]
    features_for_one_crop = tf.convert_to_tensor(features_for_one_crop)

    lx = (x[0] + x[3]) / 2
    ly = (y[0] + y[3]) / 2

    rx = (x[1] + x[2]) / 2
    ry = (y[1] + y[2]) / 2

    angle = tf.math.atan2(ry - ly, rx - lx)
    return tfa.image.rotate(features_for_one_crop, -angle, interpolation='BILINEAR')

class Encoder(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        classes = [1, 2*4]
        self.num_classes = sum(classes)
        self.max_word_len = params.max_word_len

        if params.model_name.startswith('darknet'):
            self.body = darknet.DarknetBody(params)
        elif params.model_name.startswith('gated_conv'):
            self.body = features.FeatureExtractor(params)
        elif params.model_name.startswith('efficientnet-'):
            self.body = efn_encoder.EfnBody(params)
        else:
            raise NameError('unsupported model name {}'.format(params.model_name))

        self.head = Head(params, classes)

        self.rnn_features = ConcatFeatures(params)
        self.rnn_layer = attention.RNNLayer(params, 256, self.max_word_len, params.dictionary_size, params.pad_value, cell='lstm')

        self.output_sizes = None
        self.rnn_features_scale = None

    def pick_features_for_single_image(self, features, picked_features, written, poly_single_image):
        # channels_last only
        channels = features.shape[-1]

        feature_height = tf.shape(features)[0]
        feature_width = tf.shape(features)[1]


        for poly_single in poly_single_image:
            # [4, 2] - > [4], [4]
            x = poly_single[..., 0]
            y = poly_single[..., 1]

            x = tf.maximum(x, 0.)
            y = tf.maximum(y, 0.)

            xmin = tf.math.reduce_min(x, axis=0)
            ymin = tf.math.reduce_min(y, axis=0)
            xmax = tf.math.reduce_max(x, axis=0)
            ymax = tf.math.reduce_max(y, axis=0)

            xmin = tf.cast(xmin, tf.int32)
            xmax = tf.cast(xmax, tf.int32)
            ymin = tf.cast(ymin, tf.int32)
            ymax = tf.cast(ymax, tf.int32)

            xmax = tf.minimum(xmax, feature_width)
            ymax = tf.minimum(ymax, feature_height)

            features_for_one_crop = run_crop_and_rotation(features, x, y, xmin, ymin, xmax, ymax)

            features_for_one_crop = preprocess.pad_resize_image(features_for_one_crop, [8, 8])
            features_for_one_crop.set_shape([8, 8, channels])

            picked_features = picked_features.write(written, features_for_one_crop)
            written += 1

        return picked_features, written

    def rnn_inference_from_picked_features(self, picked_features, true_words, true_lengths, training):
        cropped_features = picked_features.stack()

        rnn_outputs = self.rnn_layer(cropped_features, true_words, true_lengths, training)

        rnn_outputs = tf.concat(rnn_outputs, axis=-1)
        return rnn_outputs

    def rnn_inference(self, rnn_features, word_obj_mask, poly, true_words, true_lengths, anchors_all, training):
        picked_features = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        written = 0

        batch_index = tf.range(tf.shape(rnn_features)[0])
        for idx in batch_index:
            # D = ~14k
            # [B, D, 4, 2] -> [D, 4, 2]
            poly_single_image = poly[idx, ...]

            # [B, 60, 60, C] -> [60, 60, C]
            features = rnn_features[idx, ...]

            # [B, D] -> [D]
            word_mask_single = word_obj_mask[idx]

            # N = ~10
            # [D, 4, 2] -> [N, 4, 2]
            poly_single_image = tf.boolean_mask(poly_single_image, word_mask_single)

            best_anchors = tf.boolean_mask(anchors_all[..., :2], word_mask_single)
            best_anchors = tf.expand_dims(best_anchors, 1)
            best_anchors = tf.tile(best_anchors, [1, 4, 1])
            poly_single_image = poly_single_image + best_anchors

            poly_single_image = poly_single_image * self.rnn_features_scale
            picked_features, written = self.pick_features_for_single_image(features, picked_features, written, poly_single_image)

        rnn_outputs = self.rnn_inference_from_picked_features(picked_features, true_words, true_lengths, training)
        return rnn_outputs

    def rnn_inference_from_true_values(self, rnn_features, word_obj_mask, true_word_poly, true_words, true_lengths, anchors_all, training):
        #poly = class_outputs[..., 1 : 1 + 4*2]
        poly = true_word_poly

        true_words_flat = tf.boolean_mask(true_words, word_obj_mask)
        true_lens_flat = tf.boolean_mask(true_lengths, word_obj_mask)

        poly = tf.reshape(poly, [-1, tf.shape(poly)[1], 4, 2])

        rnn_outputs = self.rnn_inference(rnn_features, word_obj_mask, poly, true_words_flat, true_lens_flat, anchors_all, training)
        return rnn_outputs

    # word mask is per anchor, i.e. this is true word_obj
    def call(self, inputs, training):
        l = self.body(inputs, training)
        head_outputs = self.head(l[0], l[1], l[2], training)

        self.output_sizes = [tf.shape(o)[1] for o in head_outputs]

        batch_size = tf.shape(inputs)[0]
        outputs = []
        for head_output in head_outputs:
            head_output_flat = tf.reshape(head_output, [batch_size, -1, self.num_classes])
            outputs.append(head_output_flat)

        class_outputs = tf.concat(outputs, axis=1)

        rnn_features = self.rnn_features(l[0], l[1], l[2])
        self.rnn_features_scale = tf.cast(tf.shape(rnn_features)[1], tf.float32) / tf.cast(tf.shape(inputs)[1], tf.float32)

        #logger.info('inputs: {}, outputs: {}, l: {}, rnn_features: {}'.format(inputs.shape, class_outputs.shape, [x.shape.as_list() for x in l], rnn_features.shape))

        return class_outputs, rnn_features

def create_params(model_name, max_word_len, dictionary_size, pad_value):
    data_format='channels_last'

    if data_format == 'channels_first':
        raise NameError('unsupported data format {}'.format(data_format))

        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = 3
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
        'max_word_len': max_word_len,
        'pad_value': pad_value,
    }

    params = GlobalParams(**params)

    return params


def create_model(model_name, max_word_len, dictionary_size, pad_value):
    params = create_params(model_name, max_word_len, dictionary_size, pad_value)
    model = Encoder(params)
    return model
