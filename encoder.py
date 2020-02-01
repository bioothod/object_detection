import collections
import logging

import tensorflow as tf
import tensorflow_addons as tfa

import anchors_gen
import attention
import darknet
import encoder_efficientnet as efn_encoder
import feature_gated_conv as gated_conv
import preprocess

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout',
    'dictionary_size', 'max_sequence_len', 'pad_value',
    'image_size', 'num_anchors',
    'dtype'
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

class ObjectDetectionLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.relu_fn = params.relu_fn or local_swish

        self.outputs = []
        for n in num_outputs:
            n *= params.num_anchors

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
        self.s2_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_2")
        self.up2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

        self.s1_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s1_input = EncoderConvList(params, [256, 512, 256, 512, 256])
        self.s1_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_1")
        self.up1 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

        self.s0_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s0_input = EncoderConvList(params, [128, 256, 128, 256, 128])
        self.s0_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_0")

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

        self.raw1_upsample = darknet.DarknetUpsampling(params, 256)
        self.raw2_upsample = darknet.DarknetUpsampling(params, 512)

    def call(self, in0, in1, in2, training=True):
        x2 = self.s2_input(in2)
        x2 = self.s2_dropout(x2, training=training)
        raw2 = x2
        up2 = self.up2(x2)
        out2 = self.s2_output(x2, training=training)

        diff = up2.shape[1] - in1.shape[1]
        if diff > 0:
            in1 = self.pads[diff](in1)
        if diff < 0:
            up2 = self.pads[-diff](up2)

        x1 = tf.keras.layers.concatenate([up2, in1])
        x1 = self.s1_input(x1)
        x1 = self.s1_dropout(x1, training=training)
        raw1 = x1
        up1 = self.up1(x1)
        out1 = self.s1_output(x1, training=training)

        diff = up1.shape[1] - in0.shape[1]
        if diff > 0:
            in0 = self.pads[diff](in0)
        if diff < 0:
            up1 = self.pads[-diff](up1)

        x0 = tf.keras.layers.concatenate([up1, in0])
        x0 = self.s0_input(x0)
        x0 = self.s1_dropout(x0, training=training)
        raw0 = x0
        out0 = self.s0_output(x0, training=training)


        x = self.raw2_upsample(raw2, training=training)
        x = tf.concat([raw1, x], -1)

        x = self.raw1_upsample(x, training=training)
        x = tf.concat([raw0, x], -1)

        return [out2, out1, out0], x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.num_anchors = params.num_anchors

        classes = [1, 2*4]
        self.num_classes = sum(classes)
        self.max_sequence_len = params.max_sequence_len
        self.dictionary_size = params.dictionary_size
        self.image_size = params.image_size

        if params.model_name.startswith('darknet'):
            self.body = darknet.DarknetBody(params)
        elif params.model_name.startswith('gated_conv'):
            self.body = gated_conv.FeatureExtractor(params)
        elif params.model_name.startswith('efficientnet-'):
            self.body = efn_encoder.EfnBody(params)
        else:
            raise NameError('unsupported model name {}'.format(params.model_name))

        self.rnn_layer = attention.RNNLayer(params, 256, self.max_sequence_len, params.dictionary_size, params.pad_value, cell='lstm')
        self.head = Head(params, classes)

        self.output_sizes = None

    def rnn_inference(self, features, word_obj_mask, poly, true_words, true_lengths, anchors_all, training):
        poly = tf.boolean_mask(poly, word_obj_mask)
        true_words = tf.boolean_mask(true_words, word_obj_mask)
        true_lengths = tf.boolean_mask(true_lengths, word_obj_mask)

        best_anchors = tf.boolean_mask(anchors_all[..., :2], word_obj_mask)
        best_anchors = tf.expand_dims(best_anchors, 1)
        best_anchors = tf.tile(best_anchors, [1, 4, 1])
        poly = poly + best_anchors

        feature_size = tf.cast(tf.shape(features)[1], tf.float32)
        poly = poly * feature_size / self.image_size
        crop_size = 16
        crop_size = [crop_size, crop_size]

        bboxes = anchors_gen.polygon2bbox(poly, want_yx=True)
        bboxes /= feature_size

        batch_size = tf.shape(features)[0]
        feature_size = tf.shape(word_obj_mask)[1]
        batch_index = tf.range(batch_size, dtype=tf.int32)
        batch_index = tf.expand_dims(batch_index, 1)
        batch_index = tf.tile(batch_index, [1, feature_size])
        box_index = tf.boolean_mask(batch_index, word_obj_mask)

        selected_features = tf.image.crop_and_resize(features, bboxes, box_index, crop_size)
        return self.rnn_layer(selected_features, true_words, true_lengths, training)

        max_outputs = tf.shape(selected_features)[0]
        num_outputs = 0

        outputs = tf.TensorArray(features.dtype, size=0, dynamic_size=True, infer_shape=False)
        outputs_ar = tf.TensorArray(features.dtype, size=0, dynamic_size=True, infer_shape=False)
        written = 0

        current_batch_size = 32
        start = 0

        while num_outputs < max_outputs:
            end = tf.minimum(max_outputs - num_outputs, current_batch_size)

            tw = true_words[start:end, ...]
            tl = true_lengths[start:end, ...]
            features = selected_features[start:end, ...]

            out, out_ar = self.rnn_layer(features, tw, tl, training)
            outputs = outputs.write(written, out)
            outputs_ar = outputs_ar.write(written, out_ar)
            written += 1

            logger.info('features: {}, outputs: {}'.format(features.shape, out.shape))

            num_outputs += tf.shape(features)[0]
            start += tf.shape(features)[0]

            if max_outputs - num_outputs >= current_batch_size:
                continue
            if max_outputs == num_outputs:
                break

            current_batch_size = tf.math.pow(2, tf.cast(tf.math.log(tf.cast(max_outputs - num_outputs, tf.float32)) / tf.math.log(2.), tf.int32))

        outputs = outputs.concat()
        outputs_ar = outputs_ar.concat()

        expected_output_shape = tf.TensorShape([None, self.max_sequence_len, self.dictionary_size])
        outputs.set_shape(expected_output_shape)
        outputs_ar.set_shape(expected_output_shape)

        return outputs, outputs_ar

    def rnn_inference_from_true_values(self, raw_features, word_obj_mask, true_word_poly, true_words, true_lengths, anchors_all, training):
        #poly = class_outputs[..., 1 : 1 + 4*2]

        poly = true_word_poly
        poly = tf.reshape(poly, [-1, tf.shape(poly)[1], 4, 2])

        return self.rnn_inference(raw_features, word_obj_mask, poly, true_words, true_lengths, anchors_all, training)

    # word mask is per anchor, i.e. this is true word_obj
    def call(self, inputs, training):
        l = self.body(inputs, training)
        head_outputs, raw_features = self.head(l[0], l[1], l[2], training)

        self.output_sizes = [tf.shape(o)[1] for o in head_outputs]

        batch_size = tf.shape(inputs)[0]

        class_outputs = []

        for head_output in head_outputs:
            head_output_flat = tf.reshape(head_output, [batch_size, -1, self.num_classes])
            class_outputs.append(head_output_flat)

        class_outputs_concat = tf.concat(class_outputs, axis=1)

        #logger.info('inputs: {}, outputs: {}, l: {}, raw_features: {}'.format(inputs.shape, class_outputs.shape, [x.shape.as_list() for x in l], raw_features.shape))

        return class_outputs_concat, raw_features

def create_params(model_name, image_size, max_sequence_len, dictionary_size, pad_value, dtype):
    data_format='channels_last'

    if data_format == 'channels_first':
        raise NameError('unsupported data format {}'.format(data_format))

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
        'max_sequence_len': max_sequence_len,
        'pad_value': pad_value,
        'image_size': image_size,
        'num_anchors': anchors_gen.num_anchors,
        'dtype': dtype,
    }

    params = GlobalParams(**params)

    return params


def create_model(model_name, image_size, max_sequence_len, dictionary_size, pad_value, dtype):
    params = create_params(model_name, image_size, max_sequence_len, dictionary_size, pad_value, dtype)
    model = Encoder(params)
    return model
