import collections
import logging
import math
import typing

import tensorflow as tf

import anchors
import bndbox
import config
import efficientnet as efn
import utils

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'l2_reg_weight', 'spatial_dims', 'channel_axis',
    'd'
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
def local_swish(x):
    return x * tf.nn.sigmoid(x)

class EfnBody(tf.keras.layers.Layer):
    def __init__(self, params, model_name, **kwargs):
        super(EfnBody, self).__init__(**kwargs)

        efn_param_keys = efn.GlobalParams._fields
        efn_params = {}
        for k, v in params._asdict().items():
            if k in efn_param_keys:
                efn_params[k] = v

        self.base_model = efn.build_model(model_name, override_params=efn_params)

        self.reduction_indexes = [1, 2, 3, 4, 5]

    def call(self, inputs, training=True):
        self.endpoints = []

        outputs = self.base_model(inputs, training=training, features_only=True)

        for reduction_idx in self.reduction_indexes:
            endpoint = self.base_model.endpoints['reduction_{}'.format(reduction_idx)]
            self.endpoints.append(endpoint)

        return self.endpoints

class EfnBB(tf.keras.layers.Layer):
    def __init__(self, width, depth, num_anchors, **kwargs):
        super().__init__(**kwargs)

        self.convs = []
        for i in range(depth):
            conv = utils.ConvBlock(features=width, kernel_size=3, activation='swish', padding='same')
            self.convs.append(conv)

        self.bbox_regression = tf.keras.layers.Conv2D(num_anchors * 4,
                                                      kernel_size=3,
                                                      padding='same')

    def call(self, inputs, training=True):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)

        x = self.bbox_regression(x)

        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(x, [batch_size, -1, 4])
        return x

class EfnClassifier(tf.keras.layers.Layer):
    def __init__(self, num_features, width, depth, num_anchors, **kwargs):
        super().__init__(**kwargs)

        self.num_features = num_features

        self.convs = []
        for i in range(depth):
            conv = utils.ConvBlock(features=width, kernel_size=3, activation='swish', padding='same')
            self.convs.append(conv)

        prob = 0.01
        w_init = tf.constant_initializer(-math.log((1 - prob) / prob))
        self.cls_score = tf.keras.layers.Conv2D(num_anchors * num_features,
                                                kernel_size=3,
                                                activation='sigmoid',
                                                padding='same',
                                                bias_initializer=w_init)

    def call(self, inputs, training=True):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)

        x = self.cls_score(x)

        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(x, [batch_size, -1, self.num_features])
        return x

class EfnDet(tf.keras.Model):
    def __init__(self, params, d, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.config = config.DetConfig(d=d)
        self.anchors_config = config.AnchorsConfig()
        num_anchors = 9

        self.body = EfnBody(params, model_name=f'efficientnet-b{d:d}', name='efn')
        self.neck = utils.BiFPN(self.config.bifpn_width, self.config.bifpn_depth, name='bifpn')
        self.class_head = EfnClassifier(num_features=num_classes, width=self.config.bifpn_width, depth=self.config.bifpn_depth, num_anchors=num_anchors, name='class_head')
        self.bb_head = EfnBB(width=self.config.bifpn_width, depth=self.config.bifpn_depth, num_anchors=num_anchors, name='bb_head')

        self.anchors_gen = [anchors.AnchorGenerator(
            size=self.anchors_config.sizes[i - 3],
            aspect_ratios=self.anchors_config.ratios,
            stride=self.anchors_config.strides[i - 3]
        ) for i in range(3, 8)] # 3 to 7 pyramid levels

    def call(self,
             images: tf.Tensor,
             training: bool,
             score_threshold: float = 0.3,
             iou_threshold: float = 0.45,
             max_ret: int = 100
            ):
        backend_features = self.body(images, training=training)
        bifnp_features = self.neck(backend_features, training=training)
        bboxes = [self.bb_head(bf, training=training) for bf in bifnp_features]
        class_scores = [self.class_head(bf, training=training) for bf in bifnp_features]

        bboxes = tf.concat(bboxes, axis=1)
        class_scores = tf.concat(class_scores, axis=1)

        if training:
            return bboxes, class_scores

        im_shape = tf.shape(images)
        batch_size, h, w = im_shape[0], im_shape[1], im_shape[2]

        # Create the anchors
        anchors = [g(f[0].shape) for g, f in zip(self.anchors_gen, bifnp_features)]
        anchors = tf.concat(anchors, axis=0)

        # Tile anchors over batches, so they can be regressed
        anchors = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])

        boxes = bndbox.regress_bndboxes(anchors, bboxes)
        boxes = bndbox.clip_boxes(boxes, [h, w])
        boxes, scores, labels = bndbox.nms(boxes, class_scores, score_threshold=score_threshold, iou_threshold=iou_threshold, max_ret=max_ret)

        return boxes, scores, labels

def create_model(d, num_classes, name='efndet'):
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
    }

    params = GlobalParams(**params)

    model = EfnDet(params, d, num_classes, name=name)
    return model
