import collections
import logging
import re

import numpy as np
import tensorflow as tf

import anchor
import efficientnet as efn

logger = logging.getLogger('detection')
logger.setLevel(logging.INFO)

def local_swish(x):
    return x * tf.nn.sigmoid(x)

class StdConv(tf.keras.layers.Layer):
    def __init__(self, global_params, num_filters, strides=2, dilation_rate=1, dropout_rate=0.2, padding='same', **kwargs):
        super(StdConv, self).__init__(**kwargs)

        self.num_filters = num_filters
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.padding = padding

        self.relu_fn = global_params.relu_fn or tf.nn.swish
        self.batch_norm_momentum = global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.data_format = global_params.data_format
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_dims = [2, 3]
        else:
            self.channel_axis = -1
            self.spatial_dims = [1, 2]

    def build(self, input_shape):
        self.c0 = tf.keras.layers.Conv2D(
            input_shape=input_shape,
            filters=self.num_filters,
            kernel_size=1,
            strides=1,
            data_format=self.data_format,
            activation=self.relu_fn,
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=efn.conv_kernel_initializer,
            padding=self.padding)
        self.c1= tf.keras.layers.Conv2D(
            input_shape=input_shape,
            filters=self.num_filters*2,
            kernel_size=3,
            strides=self.strides,
            activation=self.relu_fn,
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            data_format=self.data_format,
            kernel_initializer=efn.conv_kernel_initializer,
            padding=self.padding)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.channel_axis,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=True):
        x = self.c0(inputs)
        x = self.c1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training)

        return x

class OutConv(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes, **kwargs):
        super(OutConv, self).__init__(**kwargs)

        self.k = k
        self.num_classes = num_classes

        self.data_format = global_params.data_format
        self.relu_fn = global_params.relu_fn or tf.nn.swish

    def build(self, input_shape):
        self.class_out = tf.keras.layers.Conv2D(
            input_shape=input_shape,
            filters=self.num_classes*self.k,
            kernel_size=3,
            strides=1,
            activation=self.relu_fn,
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            data_format=self.data_format,
            padding='same')

        self.loc_out = tf.keras.layers.Conv2D(
            input_shape=input_shape,
            filters=4*self.k,
            kernel_size=3,
            strides=1,
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            data_format=self.data_format,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            padding='same',
            activation='relu')

    def flatten_anchors(self, x):
        return tf.reshape(x, [-1, x.shape[1] * x.shape[2] * self.k, int(x.shape[3] / self.k)])

    def call(self, inputs, training=True):
        coords = self.loc_out(inputs)
        classes = self.class_out(inputs)

        flatten_coords = self.flatten_anchors(coords)
        flatten_classes = self.flatten_anchors(classes)

        #logger.info('output conv: coords: {} -> {}, classes: {} -> {}'.format(coords.shape, flatten_coords.shape, classes.shape, flatten_classes.shape))

        return [flatten_coords, flatten_classes]

class FeatureLayer(tf.keras.layers.Layer):
    def __init__(self, global_params, **kwargs):
        super(FeatureLayer, self).__init__(**kwargs)
        self.global_params = global_params
        self.relu_fn = global_params.relu_fn or tf.nn.swish

    def build(self, input_shape):
        self.dropout = tf.keras.layers.Dropout(0.25)
        self.sc0 = StdConv(self.global_params, 512, strides=2)
        self.sc1 = StdConv(self.global_params, 256, strides=1)

    def call(self, inputs, training=True):
        x = self.relu_fn(inputs)
        x = self.dropout(x, training)
        x = self.sc0(x)
        x = self.sc1(x)
        return x

class SSDHead(tf.keras.layers.Layer):
    def __init__(self, global_params, k, num_classes, **kwargs):
        super(SSDHead, self).__init__(**kwargs)

        self.k = k
        self.num_classes = num_classes

        self.global_params = global_params

    def build(self, input_shape):
        self.sc1 = StdConv(self.global_params, 512, strides=1)
        self.out = OutConv(self.global_params, self.k, self.num_classes)

    def call(self, inputs, training=True):
        x = self.sc1(inputs)
        x = self.out(x)

        return x

class EfficientNetSSD(tf.keras.Model):
    def __init__(self, model_name, num_classes, **kwargs):
        super(EfficientNetSSD, self).__init__(**kwargs)

        override_params = {
            'data_format': 'channels_last',
            'num_classes': None,
            'relu_fn': local_swish,
        }
        blocks_args, global_params, image_size = efn.get_model_params(model_name, override_params)
        self.base_model = efn.Model(blocks_args, global_params)


        self.global_params = global_params
        self.blocks_args = blocks_args
        self.relu_fn = global_params.relu_fn or tf.nn.swish

        self.image_size = image_size
        self.num_classes = num_classes

        self.cell_ratios_for_layers = [
            [1.3, 2, 3, 4.1],
            [1.3, 2, 3, 4.1],
            [1.3, 2, 3, 4.1],
            [1.3, 2, 3, 4.1],
            [1.3, 2, 3, 4.1],
            [1.3, 2, 3, 4.1],
        ]
        self.square_scales = [1, 0.5, 0.75]

        self.ssd_heads = {}
        self.ssd_meta = {}
        self.output_layer_idxs = []

        self.top_layers = []

        self.endpoints = None

        self.build_ssd()

    def build_ssd_head(self, layer_idx):
        cell_ratios = self.cell_ratios_for_layers[layer_idx]

        aspect_ratios = []
        for ratio in cell_ratios:
            ratio = np.sqrt(ratio)
            aspect_ratios += [(1/ratio, ratio), (ratio, 1/ratio)]

        for scale in self.square_scales:
            aspect_ratios += [(scale, scale)]

        shifts2d = [(0, 0)]

        num_anchors_per_output = len(aspect_ratios) * len(shifts2d)

        ssd_head = SSDHead(self.global_params, num_anchors_per_output, self.num_classes)

        logger.info('ssd: layer_idx: {}, num_anchors_per_output: {}, aspect_ratios: {}'.format(layer_idx, num_anchors_per_output, len(aspect_ratios)))

        return ssd_head, (num_anchors_per_output, aspect_ratios, shifts2d)

    def build_ssd(self):
        reduction_idx = 0
        reduction_blocks = {}
        reduction_skip = 2
        top_layers = 3

        for idx, block in enumerate(self.base_model._blocks):
            if ((idx == len(self.base_model._blocks) - 1) or self.base_model._blocks[idx + 1].block_args().strides[0] > 1):
                reduction_idx += 1

                if reduction_idx <= reduction_skip:
                    continue
                reduction_blocks[reduction_idx] = block

        logger.info('reduction blocks: {}, reduction_skip: {}, reduction indexes: {}'.format(len(reduction_blocks), reduction_skip, reduction_blocks.keys()))

        if top_layers + len(reduction_blocks) != len(self.cell_ratios_for_layers):
            logger.critical('incorrect number of blocks: cell ratios: {}, must be equal to sum: reduction_blocks: {}, top_layers: {}'.format(
                len(self.cell_ratios_for_layers),
                len(reduction_blocks), top_layers))
            exit(-1)

        last_reduction_idx = None
        for reduction_idx, block in reduction_blocks.items():
            layer_idx = len(self.ssd_heads)

            head, meta = self.build_ssd_head(layer_idx)
            self.ssd_heads[reduction_idx] = head
            self.ssd_meta[reduction_idx] = meta

            self.output_layer_idxs.append(reduction_idx)

            last_reduction_idx = reduction_idx

        for top_idx in range(top_layers):
            layer_idx = len(self.ssd_heads)
            last_reduction_idx += 1

            top_layer = FeatureLayer(self.global_params)
            self.top_layers.append(top_layer)

            head, meta = self.build_ssd_head(layer_idx)
            self.ssd_heads[last_reduction_idx] = head
            self.ssd_meta[last_reduction_idx] = meta

            self.output_layer_idxs.append(last_reduction_idx)


    def call(self, inputs, training=True):
        self.endpoints = []

        ssd_stack_coords = []
        ssd_stack_classes = []

        outputs = self.base_model(inputs, training=training, features_only=True)

        last_reduction_idx = None
        for ename, endpoint in self.base_model.endpoints.items():
            m = re.match('reduction_(\d+)$', ename)
            if m is None:
                continue

            reduction_idx = int(m.group(1))

            ssd_head = self.ssd_heads.get(reduction_idx)
            if ssd_head is not None:
                coords, classes = ssd_head(endpoint, training=training)

                ssd_stack_coords.append(coords)
                ssd_stack_classes.append(classes)

                self.endpoints.append((coords, classes))
                last_reduction_idx = reduction_idx

        for top_idx, top_block in enumerate(self.top_layers):
            top_idx += last_reduction_idx + 1

            outputs = top_block(outputs, training=training)

            ssd_head = self.ssd_heads.get(top_idx)
            if ssd_head is not None:
                coords, classes = ssd_head(outputs, training=training)

                ssd_stack_coords.append(coords)
                ssd_stack_classes.append(classes)

                self.endpoints.append((coords, classes))

        output_classes = tf.concat(ssd_stack_classes, axis=1)
        output_coords = tf.concat(ssd_stack_coords, axis=1)

        return output_coords, output_classes


def create_model(dtype, model_name, num_classes):
    model = EfficientNetSSD(model_name, num_classes)

    inputs = tf.keras.layers.Input(shape=(model.image_size, model.image_size, 3))
    model(inputs, training=True)

    logger.info('called model with dummy input to determine output shapes to build anchors')

    anchor_boxes = []
    anchor_areas = []

    num_layers = len(model.output_layer_idxs)

    for layer_idx, (reduction_idx, endpoint) in enumerate(zip(model.output_layer_idxs, model.endpoints)):
        coords, classes = endpoint

        m = model.ssd_meta.get(reduction_idx)
        num_anchors_per_output, aspect_ratios, shifts2d = m

        layer_size = int(np.sqrt(classes.shape[1] / num_anchors_per_output))

        min_scale = 0.1
        max_scale = 0.9

        layer_scale = min_scale + (max_scale - min_scale) * layer_idx / (len(model.output_layer_idxs) - 1)

        anchor_boxes_for_layer, anchor_areas_for_layer = anchor.create_anchors_for_layer(model.image_size, layer_size, layer_scale, aspect_ratios, shifts2d)
        anchor_boxes += anchor_boxes_for_layer
        anchor_areas += anchor_areas_for_layer

        logger.info('ssd_head: layer_idx: {}, layer_scale: {:.2f}, layer_size: {}, num_anchors_per_output: {}, anchors: {}/{}'.format(
            layer_idx, layer_scale, layer_size, num_anchors_per_output, len(anchor_boxes_for_layer), len(anchor_boxes)))

    anchor_boxes = np.array(anchor_boxes)
    anchor_areas = np.array(anchor_areas)

    logger.info('model: model_name: {}, anchor_boxes: {}'.format(model_name, anchor_boxes.shape))

    return model, anchor_boxes, anchor_areas
