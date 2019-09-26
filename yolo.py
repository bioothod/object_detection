import collections
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'relu_fn', 'l2_reg_weight', 'channel_axis', 'spatial_dims'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

class DarknetConv(tf.keras.layers.Layer):
    def __init__(self, params, num_features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', **kwargs):
        super(DarknetConv, self).__init__(**kwargs)
        self.params = params
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
                self.num_features,
                input_shape=input_shape,
                kernel_size=self.kernel_size,
                strides=self.strides,
                data_format=self.params.data_format,
                use_bias=False,
                padding=self.padding,
                kernel_initializer='glorot_uniform')

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.params.channel_axis,
            momentum=self.params.batch_norm_momentum,
            epsilon=self.params.batch_norm_epsilon)

    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x

class DarknetConvPool(tf.keras.layers.Layer):
    def __init__(self, params, num_features, **kwargs):
        super(DarknetConvPool, self).__init__(**kwargs)
        self.params = params
        self.num_features = num_features

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), input_shape=input_shape)
        self.conv = DarknetConv(self.params,
                self.num_features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='VALID')

    def call(self, inputs, training):
        x = self.pad(inputs)
        x = self.conv(x, training)
        return x

class DarknetResidual(tf.keras.layers.Layer):
    def __init__(self, params, filters, **kwargs):
        super(DarknetResidual, self).__init__(**kwargs)
        self.filters = filters
        self.params = params

    def build(self, input_shape):
        self.conv0 = DarknetConv(self.params, self.filters[0], kernel_size=(1, 1), strides=(1, 1), padding='SAME', input_shape=input_shape)
        self.conv1 = DarknetConv(self.params, self.filters[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    def call(self, inputs, training):
        x = self.conv0(inputs, training)
        x = self.conv1(x, training)

        x += inputs
        return x

class DarknetBody(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(DarknetBody, self).__init__(**kwargs)
        
        # (256, 256, 3)
        self.l0a = DarknetConv(params, 32, name="l0")
        self.l0_pool = DarknetConvPool(params, 64, name="l0_pool")

        # (128, 128, 64)
        self.l1a = DarknetResidual(params, [32, 64], name="l1")
        self.l1_pool = DarknetConvPool(params, 128, name="l1_pool")

        # (64, 64, 128)
        self.l2a = DarknetResidual(params, [64, 128], name="l2a")
        self.l2b = DarknetResidual(params, [64, 128], name="l2b")
        self.l2_pool = DarknetConvPool(params, 256, name="l2_pool")

        # (32, 32, 256)
        self.l3a = DarknetResidual(params, [128, 256], name="l3a")
        self.l3b = DarknetResidual(params, [128, 256], name="l3b")
        self.l3c = DarknetResidual(params, [128, 256], name="l3c")
        self.l3d = DarknetResidual(params, [128, 256], name="l3d")
        self.l3e = DarknetResidual(params, [128, 256], name="l3e")
        self.l3f = DarknetResidual(params, [128, 256], name="l3f")
        self.l3g = DarknetResidual(params, [128, 256], name="l3g")
        self.l3h = DarknetResidual(params, [128, 256], name="l3h")
        self.l3_pool = DarknetConvPool(params, 512, name="l3_pool")
        
        # (16, 16, 512)
        self.l4a = DarknetResidual(params, [256, 512], name="l4a")
        self.l4b = DarknetResidual(params, [256, 512], name="l4b")
        self.l4c = DarknetResidual(params, [256, 512], name="l4c")
        self.l4d = DarknetResidual(params, [256, 512], name="l4d")
        self.l4e = DarknetResidual(params, [256, 512], name="l4e")
        self.l4f = DarknetResidual(params, [256, 512], name="l4f")
        self.l4g = DarknetResidual(params, [256, 512], name="l4g")
        self.l4h = DarknetResidual(params, [256, 512], name="l4h")
        self.l4_pool = DarknetConvPool(params, 1024, name="l4_pool")

        # (8, 8, 1024)
        self.l5a = DarknetResidual(params, [512, 1024], name="l5a")
        self.l5b = DarknetResidual(params, [512, 1024], name="l5b")
        self.l5c = DarknetResidual(params, [512, 1024], name="l5c")
        self.l5d = DarknetResidual(params, [512, 1024], name="l5d")
        
    def call(self, inputs, training):
        x = self.l0a(inputs, training)
        x = self.l0_pool(x, training)

        x = self.l1a(x, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2b(x, training)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3c(x, training)
        x = self.l3d(x, training)
        x = self.l3e(x, training)
        x = self.l3f(x, training)
        x = self.l3g(x, training)
        x = self.l3h(x, training)
        output_l3 = x
        x = self.l3_pool(x, training)

        x = self.l4a(x, training)
        x = self.l4b(x, training)
        x = self.l4c(x, training)
        x = self.l4d(x, training)
        x = self.l4e(x, training)
        x = self.l4f(x, training)
        x = self.l4g(x, training)
        x = self.l4h(x, training)
        output_l4 = x
        x = self.l4_pool(x, training)

        x = self.l5a(x, training)
        x = self.l5b(x, training)
        x = self.l5c(x, training)
        x = self.l5d(x, training)
        output_l5 = x
        return output_l3, output_l4, output_l5

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

        self.conv = DarknetConv(params, num_features, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.upsampling = tf.keras.layers.UpSampling2D(2)

    def call(self, input_tensor, training):
        x = self.conv(input_tensor, training)
        x = self.upsampling(x)
        return x

class DarknetHead(tf.keras.layers.Layer):
    def __init__(self, params, num_classes, **kwargs):
        super(DarknetHead, self).__init__(**kwargs)
        num_features = 3 * (num_classes+1+4)
        
        self.stage5_conv5 = DarknetConv5(params, [512, 1024, 512, 1024, 512])
        self.stage5_conv2 = DarknetConv2(params, [1024, num_features], name="detection_layer_1")
        self.stage5_upsampling = DarknetUpsampling(params, 256)

        self.stage4_conv5 = DarknetConv5(params, [256, 512, 256, 512, 256])
        self.stage4_conv2 = DarknetConv2(params, [512, num_features], name="detection_layer_2")
        self.stage4_upsampling = DarknetUpsampling(params, 128)

        self.stage3_conv5 = DarknetConv5(params, [128, 256, 128, 256, 128])
        self.stage3_conv2 = DarknetConv2(params, [256, num_features], name="detection_layer_3")

    def call(self, stage3_in, stage4_in, stage5_in, training):
        x = self.stage5_conv5(stage5_in, training)
        stage5_output = self.stage5_conv2(x, training)

        x = self.stage5_upsampling(x, training)
        x = tf.keras.layers.concatenate([x, stage4_in])
        x = self.stage4_conv5(x, training)
        stage4_output = self.stage4_conv2(x, training)

        x = self.stage4_upsampling(x, training)
        x = tf.keras.layers.concatenate([x, stage3_in])
        x = self.stage3_conv5(x, training)
        stage3_output = self.stage3_conv2(x, training)

        return stage5_output, stage4_output, stage3_output


class Yolo3(tf.keras.Model):
    def __init__(self, params, num_classes, **kwargs):
        super(Yolo3, self).__init__(**kwargs)

        self.body = DarknetBody(params)
        self.head = DarknetHead(params, num_classes)

    def load_darknet_params(self, weights_file, skip_detect_layer=False):
        weight_reader = WeightReader(weights_file)
        weight_reader.load_weights(self, skip_detect_layer)

    def call(self, input_tensor, training):
        s3, s4, s5 = self.body(input_tensor, training)
        f5, f4, f3 = self.head(s3, s4, s5, training)
        return f5, f4, f3

def local_relu(x):
    return x * tf.nn.sigmoid(x)

def create_model(num_classes):
    data_format='channels_last'

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        data_format=data_format,
        channel_axis=channel_axis,
        spatial_dims=spatial_dims,
        relu_fn=local_relu)


    model = Yolo3(params, num_classes)
    return model

def create_anchors():
    anchors_dict = {
        '0': [
            [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
            416,
        ],
        '1': [
            [(17, 18), (28, 24), (36, 34), (42, 44), (56, 51), (72, 66), (90, 95), (92, 154), (139, 281)],
            352,
        ],
    }

    anchors, image_size = anchors_dict["0"]
    # _,_,W,H format
    anchors = np.array(anchors, dtype=np.float32)
    areas = anchors[:, 0] * anchors[:, 1]

    return anchors, areas, image_size
