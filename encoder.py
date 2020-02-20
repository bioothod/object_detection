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
import stn

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout',
    'dictionary_size', 'max_sequence_len', 'pad_value',
    'image_size', 'num_anchors', 'crop_size',
    'dtype'
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
def local_swish(x):
    return x * tf.nn.sigmoid(x)

class EncoderConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
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
                padding=self.padding)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.bn(inputs, training)
        x = self.conv(x)
        x = self.relu_fn(x)

        return x

class EncoderConvList(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super().__init__(**kwargs)

        self.convs = []

        kernel_size = 1
        for num_filters in filters:
            conv = EncoderConv(params, num_filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='SAME')
            self.convs.append(conv)

            if kernel_size == 3:
                kernel_size = 1
            else:
                kernel_size = 3

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
                use_bias=False,
                padding='SAME')

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
        self.s2_input = EncoderConvList(params, [512])
        self.s2_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_2")
        self.up2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

        self.s1_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s1_input = EncoderConvList(params, [256])
        self.s1_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_1")
        self.up1 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')

        self.s0_dropout = tf.keras.layers.Dropout(params.spatial_dropout)
        self.s0_input = EncoderConvList(params, [128])
        self.s0_output = ObjectDetectionLayer(params, num_classes, name="detection_layer_0")

        self.pads = [None]
        for pad in [1, 2, 3, 4, 5, 6]:
            self.pads.append(tf.keras.layers.ZeroPadding2D(((0, pad), (0, pad))))

    def call(self, in0, in1, in2, training=True):
        x2 = self.s2_input(in2)
        x2 = self.s2_dropout(x2, training=training)
        up2 = self.up2(x2)
        up2 = tf.cast(up2, in0.dtype)
        out2 = self.s2_output(x2, training=training)

        diff = up2.shape[1] - in1.shape[1]
        if diff > 0:
            in1 = self.pads[diff](in1)
        if diff < 0:
            up2 = self.pads[-diff](up2)

        x1 = tf.concat([up2, in1], -1)
        x1 = self.s1_input(x1)
        x1 = self.s1_dropout(x1, training=training)
        up1 = self.up1(x1)
        up1 = tf.cast(up1, in0.dtype)
        out1 = self.s1_output(x1, training=training)

        diff = up1.shape[1] - in0.shape[1]
        if diff > 0:
            in0 = self.pads[diff](in0)
        if diff < 0:
            up1 = self.pads[-diff](up1)

        x0 = tf.concat([up1, in0], -1)
        x0 = self.s0_input(x0)
        x0 = self.s1_dropout(x0, training=training)
        out0 = self.s0_output(x0, training=training)

        return [out2, out1, out0]

def gaussian_mask(mu, sigma, dim, dim_tile):
    r = tf.cast(tf.range(dim), tf.float32)
    r = tf.expand_dims(r, 0)
    r = tf.tile(r, [tf.shape(mu)[0], 1])

    #sigma = tf.expand_dims(sigma, -1)
    #mu = tf.expand_dims(mu, -1)

    centers = r - mu
    mask = tf.exp(-.5 * tf.square(centers / sigma))
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, dim_tile])
    #mask = mask / (tf.reduce_sum(mask, 1, keepdims=True) + 1e-8)
    return mask

def generate_gaussian_mask(image, mask_params):
    h = tf.shape(image)[1]
    w = tf.shape(image)[2]
    c = tf.shape(image)[3]

    mu_h, sigma_h, mu_w, sigma_w = tf.split(mask_params, 4, axis=-1)

    mask_h = gaussian_mask(mu_h, sigma_h, h, w)
    mask_w = gaussian_mask(mu_w, sigma_w, w, h)
    mask_w = tf.transpose(mask_w, (0, 2, 1))

    mask = mask_h * mask_w

    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 1, c])

    new_image = image * mask

    return new_image

class Encoder(tf.keras.Model):
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

        self.crop_size = params.crop_size

        self.rnn_layer = attention.RNNLayer(params, self.max_sequence_len, params.dictionary_size, params.pad_value)
        self.head = Head(params, classes)

        self.output_sizes = None

    def rnn_inference(self, features_full, word_obj_mask_full, poly_full, true_words_full, true_lengths_full, anchors_all, training):
        dtype = features_full.dtype

        true_words = tf.boolean_mask(true_words_full, word_obj_mask_full)
        true_lengths = tf.boolean_mask(true_lengths_full, word_obj_mask_full)

        logger.info('RNN features: {}, crop_size: {}'.format(features_full.shape, list(self.crop_size)))
        feature_size = tf.cast(tf.shape(features_full)[1], dtype)

        batch_size = tf.shape(features_full)[0]

        selected_features = tf.TensorArray(dtype, size=0, dynamic_size=True)
        written = 0

        for idx in tf.range(batch_size):
            poly = poly_full[idx, ...]
            word_obj_mask = word_obj_mask_full[idx, ...]
            features = features_full[idx, ...]

            poly = tf.boolean_mask(poly, word_obj_mask)

            best_anchors = tf.boolean_mask(anchors_all[..., :2], word_obj_mask)
            best_anchors = tf.expand_dims(best_anchors, 1)
            best_anchors = tf.tile(best_anchors, [1, 4, 1])
            poly = poly + best_anchors

            poly = poly * feature_size / self.image_size

            x = poly[..., 0]
            y = poly[..., 1]
            lx = (x[:, 0] + x[:, 3]) / 2
            ly = (y[:, 0] + y[:, 3]) / 2

            rx = (x[:, 1] + x[:, 2]) / 2
            ry = (y[:, 1] + y[:, 2]) / 2

            angles = tf.math.atan2(ry - ly, rx - lx)

            x0 = x[:, 0]
            y0 = y[:, 0]
            x1 = x[:, 1]
            y1 = y[:, 1]
            x2 = x[:, 2]
            y2 = y[:, 2]
            x3 = x[:, 3]
            y3 = y[:, 3]

            h0 = (x0 - x3)**2 + (y0 - y3)**2
            h1 = (x1 - x2)**2 + (y1 - y2)**2
            h = tf.math.sqrt(tf.maximum(h0, h1)) + 2

            w0 = (x0 - x1)**2 + (y0 - y1)**2
            w1 = (x2 - x3)**2 + (y2 - y3)**2
            w = tf.math.sqrt(tf.maximum(w0, w1)) + 2

            sy = h / self.crop_size[0]
            sx = w / self.crop_size[1]
            scale = tf.maximum(sx, sy)

            z = tf.zeros_like(x0)
            o = tf.ones_like(x0)

            def tm(t):
                t = tf.convert_to_tensor(t)
                t = tf.transpose(t, [2, 0, 1])
                return t

            def const(x):
                return o * x

            t0 = tm([[o, z, x0-1], [z, o, y0-1], [z, z, o]])
            t1 = tm([[tf.cos(angles), -tf.sin(angles), z], [tf.sin(angles), tf.cos(angles), z], [z, z, o]])
            t2 = tm([[scale, z, z], [z, scale, z], [z, z, o]])

            transforms = t0 @ t1 @ t2
            transforms = transforms[:, :2, :]

            xsize = w / scale
            ysize = h / scale

            mu_w = xsize / 2
            mu_h = ysize / 2
            sigma_w = xsize / 4
            sigma_h = ysize / 4
            mask_params = tf.stack([mu_h, sigma_h, mu_w, sigma_w], 1)

            features = tf.expand_dims(features, 0)
            num_crops = tf.shape(transforms)[0]
            for crop_idx in tf.range(num_crops):
                t = transforms[crop_idx, ...]
                t = tf.expand_dims(t, 0)

                mp = tf.expand_dims(mask_params[crop_idx, ...], 0)

                cropped_features = stn.spatial_transformer_network(features, t, out_dims=self.crop_size)
                cropped_features = generate_gaussian_mask(cropped_features, mp)

                selected_features = selected_features.write(written, cropped_features)
                written += 1

        selected_features = selected_features.concat()

        return self.rnn_layer(selected_features, true_words, true_lengths, training)


    def rnn_inference_from_true_values(self, class_outputs, raw_features, word_obj_mask, true_word_poly, true_words, true_lengths, anchors_all, training, use_predicted_polys):
        if use_predicted_polys:
            poly = class_outputs[..., 1 : 1 + 4*2]
        else:
            poly = true_word_poly

        poly = tf.reshape(poly, [-1, tf.shape(poly)[1], 4, 2])

        return self.rnn_inference(raw_features, word_obj_mask, poly, true_words, true_lengths, anchors_all, training)

    # word mask is per anchor, i.e. this is true word_obj
    def call(self, inputs, training):
        l, raw_features = self.body(inputs, training)
        head_outputs = self.head(l[0], l[1], l[2], training)

        self.output_sizes = [tf.shape(o)[1] for o in head_outputs]

        batch_size = tf.shape(inputs)[0]

        class_outputs = []

        for head_output in head_outputs:
            head_output_flat = tf.reshape(head_output, [batch_size, -1, self.num_classes])
            class_outputs.append(head_output_flat)

        class_outputs_concat = tf.concat(class_outputs, axis=1)

        #logger.info('inputs: {}, outputs: {}, l: {}, raw_features: {}'.format(inputs.shape, class_outputs.shape, [x.shape.as_list() for x in l], raw_features.shape))

        return class_outputs_concat, raw_features

def create_params(model_name, image_size, crop_size, max_sequence_len, dictionary_size, pad_value, dtype):
    data_format='channels_last'

    crop = crop_size.split('x')[:2]
    crop_size = [int(c) for c in crop]

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
        'crop_size': crop_size,
    }

    params = GlobalParams(**params)

    return params


def create_model(model_name, image_size, crop_size, max_sequence_len, dictionary_size, pad_value, dtype):
    params = create_params(model_name, image_size, crop_size, max_sequence_len, dictionary_size, pad_value, dtype)
    model = Encoder(params)
    return model
