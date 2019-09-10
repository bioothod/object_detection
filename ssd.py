import collections
import logging

import numpy as np
import tensorflow as tf

import efficientnet.tfkeras as efn

import anchor

logger = logging.getLogger('detection')
logger.setLevel(logging.INFO)

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'relu_fn', 'l2_reg_weight'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

class StdConv(tf.keras.layers.Layer):
    def __init__(self, global_params, num_filters, strides=2, dilation_rate=1, dropout_rate=0.1):
        super(StdConv, self).__init__()

        self.num_filters = num_filters
        self.strides = strides

        self.l2_reg_weight = global_params.l2_reg_weight
        self.batch_norm_momentum = global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.data_format = global_params.data_format
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_dims = [2, 3]
        else:
            self.channel_axis = -1
            self.spatial_dims = [1, 2]

        self._build()

    def _build(self):
        self.c0 = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=3,
            strides=self.strides,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            bias_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            padding='same',
            use_bias=False)

        self.bn0 = tf.keras.layers.BatchNormalization(
            axis=self.channel_axis,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon)

        self.dropout0 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=True):
        x = self.c0(inputs)
        x = self.bn0(x, training=training)
        x = self.dropout0(x)

        return x

class OutConv(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes):
        super(OutConv, self).__init__()

        self.k = k
        self.num_classes = num_classes

        self.data_format = global_params.data_format
        self.l2_reg_weight = global_params.l2_reg_weight

        self._build()

    def _build(self):
        self.class_out = tf.keras.layers.Conv2D(
            filters=self.num_classes*self.k,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            bias_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            padding='same')

        self.loc_out = tf.keras.layers.Conv2D(
            filters=4*self.k,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            kernel_initializer=conv_kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            bias_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight),
            padding='same',
            activation='relu')

    def flatten_anchors(self, x):
        return tf.reshape(x, [-1, x.shape[1] * x.shape[2] * self.k, x.shape[3] // self.k])

    def call(self, inputs, training=True):
        coords = self.loc_out(inputs)
        classes = self.class_out(inputs)

        flatten_coords = self.flatten_anchors(coords)
        flatten_classes = self.flatten_anchors(classes)

        #logger.info('output conv: coords: {} -> {}, classes: {} -> {}'.format(coords.shape, flatten_coords.shape, classes.shape, flatten_classes.shape))

        return [flatten_coords, flatten_classes]

class SSD_Head(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes, top_strides, name=None):
        super(SSD_Head, self).__init__(name=name)

        self.k = k
        self.num_classes = num_classes
        self.top_strides = top_strides

        self.global_params = global_params
        self.relu_fn = global_params.relu_fn or tf.nn.swish

        self._build()

    def _build(self):
        self.dropout = tf.keras.layers.Dropout(0.25)
        self.sc0 = StdConv(self.global_params, 256, strides=1, dilation_rate=1/self.top_strides)
        self.sc1 = StdConv(self.global_params, 256, strides=1)
        self.out = OutConv(self.global_params, self.k, self.num_classes)

    def call(self, inputs, training=True):
        x = self.relu_fn(inputs)
        x = self.dropout(x)
        x = self.sc0(x)
        x = self.out(x)

        return x

def create_base_model(dtype, model_name, layer_names):
    model_map = {
        'efficientnet-b0': (efn.EfficientNetB0, 224),
        'efficientnet-b1': (efn.EfficientNetB1, 240),
        'efficientnet-b2': (efn.EfficientNetB2, 260),
        'efficientnet-b3': (efn.EfficientNetB3, 300),
        'efficientnet-b4': (efn.EfficientNetB4, 380),
        'efficientnet-b5': (efn.EfficientNetB5, 456),
        'efficientnet-b6': (efn.EfficientNetB6, 528),
        'efficientnet-b7': (efn.EfficientNetB7, 600),
    }

    base_model, image_size = model_map[model_name]
    base_model = base_model(include_top=False)
    base_model.trainable = False

    layers = [base_model.get_layer(name).output for name in layer_names]
    if len(layers) == 1:
        layers = [layers]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    #down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    features = down_stack(inputs)

    logger.info('layers: {}, features: {}'.format(layers, features))

    feature_shapes = []
    for name, l in zip(layer_names, features):
        logger.info('base model: {}, name: {}, feature layer: {}, shape: {}'.format(model_name, name, l.name, l.shape))
        feature_shapes.append([l.shape[1], l.shape[2], l.shape[3]])

    return down_stack, image_size, feature_shapes

def create_model(dtype, model_name, num_classes):
    layer_names = ['block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation']

    cell_scales = [
        [1.3, 2, 3],
        [1.3, 2, 3, 4.1],
        [1.3, 2, 3, 4.1, 4.6],
        [1.3, 2, 3, 4.5],
        [1.3, 2, 3],
    ]
    shifts = [0]
    shifts_square = []

    shifts2d = []
    for shx in shifts:
        for shy in shifts:
            shifts2d.append((shx, shy))

    for sh in shifts_square:
        shifts2d.append((sh, sh))

    down_stack, image_size, feature_shapes = create_base_model(dtype, model_name, layer_names)

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    features = down_stack(inputs)
    features += [None] * (len(cell_scales) - len(layer_names))

    min_scale = 0.2
    max_scale = 0.9
    num_features = len(features)

    global_params = GlobalParams(
        data_format='channels_last',
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-8,
        relu_fn=tf.nn.swish,
        l2_reg_weight=1e-4,
    )

    anchor_boxes, anchor_areas = [], []
    ssd_stack_coords, ssd_stack_classes = [], []
    top_strides = 1
    last_good_features = None
    layer_index = 0
    for ft, cell_scales_for_layer in zip(features, cell_scales):
        layer_scale = min_scale + (max_scale - min_scale) * (layer_index - 1) / (num_features - 1)

        if ft is None:
            ft = last_good_features
            top_strides *= 2
        else:
            last_good_features = ft

        aspect_ratios = []
        for scale in cell_scales_for_layer:
            scale = np.sqrt(scale)
            aspect_ratios += [(scale, 1/scale), (1/scale, scale)]

        aspect_ratios += [(1, 1)]

        layer_index += 1

        num_anchors_per_output = len(aspect_ratios) * len(shifts2d)
        coords, classes = SSD_Head(global_params, num_anchors_per_output, num_classes, top_strides)(ft)

        layer_size = int(np.sqrt(classes.shape[1] / num_anchors_per_output))

        anchor_boxes_for_layer, anchor_areas_for_layer = anchor.create_anchors_for_layer(image_size, layer_size, layer_scale, aspect_ratios, shifts2d)

        anchor_boxes += anchor_boxes_for_layer
        anchor_areas += anchor_areas_for_layer

        logger.info('model: input feature: {}/{}, coords: {}, classes: {}, layer_size: {}, anchors: {}, total_anchors: {}'.format(
            ft.name, ft.shape,
            coords.shape, classes.shape,
            layer_size,
            len(anchor_boxes_for_layer), len(anchor_boxes)))

        ssd_stack_coords.append(coords)
        ssd_stack_classes.append(classes)

    if len(ssd_stack_classes) == 1:
        output_classes = ssd_stack_classes[0]
        output_coords = ssd_stack_coords[0]
    else:
        output_classes = tf.concat(ssd_stack_classes, axis=1)
        output_coords = tf.concat(ssd_stack_coords, axis=1)

    anchor_boxes = np.array(anchor_boxes)
    anchor_areas = np.array(anchor_areas)

    logger.info('model: output_coords: {}, output_classes: {}, anchors: {}'.format(output_coords, output_classes, anchor_boxes.shape[0]))

    model = tf.keras.Model(inputs=inputs, outputs=[output_coords, output_classes])
    return model, image_size, anchor_boxes, anchor_areas
