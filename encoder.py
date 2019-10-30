import collections
import logging

import tensorflow as tf

import efficientnet as efn

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold'
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
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
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

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = self.relu_fn(x)
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

class EfnText(tf.keras.layers.Layer):
    def __init__(self, params, model_name, image_size, upscaled_image_size, anchors, grid_xy, ratios, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.params = params

        self.object_model = EfnYolo(params)

        self.image_size = image_size
        self.upscaled_image_size = upscaled_image_size
        self.scale = float(self.upscaled_image_size) / float(self.image_size)

        self.grid_xy = tf.expand_dims(grid_xy, 0)
        self.ratios = tf.expand_dims(ratios, 0)
        self.ratios = tf.expand_dims(ratios, -1) # [B, N, 1]
        self.anchors_wh = tf.expand_dims(anchors[..., :2], 0)

        self.relu_fn = params.relu_fn or local_swish

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

        self.dense_y = tf.keras.layers.Dense(128, activation=self.relu_fn)
        self.logits = tf.keras.layers.Dense(128, activation=self.relu_fn)

    def gaussian_masks(mu, sigma, dim, dim_tile):
        r = tf.cast(tf.range(dim), tf.float32)
        r = tf.expand_dims(r, 0)
        r = tf.tile(r, [tf.shape(mu)[0], 1])

        sigma = tf.expand_dims(sigma, -1)
        mu = tf.expand_dims(mu, -1)

        centers = r - mu
        mask = tf.exp(-.5 * tf.square(centers / sigma))
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, dim_tile])
        return mask

    def generate_masks(params, mu_h, sigma_h, mu_w, sigma_w):
        b, h, w, c = params

        mask_h = gaussian_masks(mu_h, sigma_h, h, w)
        mask_w = gaussian_masks(mu_w, sigma_w, w, h)

        mask_w = tf.transpose(mask_w, (0, 2, 1))
        mask = mask_h * mask_w

        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 1, c])

        return mask

    def call(self, inputs, training):
        object_outputs = self.object_model(inputs, training)

        obj_score = object_outputs[..., 4]

        non_background_index = tf.where(obj_score > self.params.obj_score_threshold)
        object_outputs = tf.gather_nd(object_outputs, non_background_index)

        inputs_xy = object_outputs[..., :2]
        inputs_wh = object_outputs[..., 2:4]

        pred_box_xy = self.grid_xy + tf.sigmoid(inputs_xy)
        pred_box_xy = pred_box_xy * self.ratios
        pred_box_wh = tf.math.exp(inputs_wh) * self.anchors_wh


        sigma_w = pred_box_wh[..., 0] * self.scale
        sigma_h = pred_box_wh[..., 1] * self.scale
        mu_w = pred_box_xy[..., 0] * self.scale
        mu_h = pred_box_xy[..., 1] * self.scale

        b = tf.shape(inputs)[0]
        masks = self.generate_masks([b, self.upscaled_image_size, self.upscaled_image_size, 3], mu_h, sigma_h, mu_w, sigma_w)

        x = tf.bn(inputs, training)
        x = tf.relu_fn(x)

        x = tf.reshape(x, [-1, x.shape[-1]])
        y = self.dense_y(x)

        logits = tf.layers.dense(y, 1, activation = None)

        logits = tf.reshape(logits, [-1, input_features.shape[1], 1])
        alphas = tf.nn.softmax(logits, axis = 1)

        if reduce_output:
            att_seq = tf.reduce_sum(input_features * alphas, axis = 1)
        else:
            att_seq = input_features * alphas


        logging.info('att: input_features: {}, alphas: {}, output: {}'.format(input_features, alphas, att_seq))
        return att_seq, alphas

def create_model(model_name):
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
    }

    params = GlobalParams(**params)

    model = EfnYolo(params)
    return model

