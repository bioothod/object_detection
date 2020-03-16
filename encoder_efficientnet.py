import collections
import logging

import tensorflow as tf

import efficientnet as efn

from feature_gated_conv import BlockConvUpsampling

logger = logging.getLogger('detection')

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'data_format',
    'relu_fn',
    'spatial_dims', 'channel_axis', 'model_name',
    'obj_score_threshold', 'lstm_dropout', 'spatial_dropout'
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

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

        self.raw0_upsampling = BlockConvUpsampling(params, [256], want_upsampling=False)
        self.raw1_upsampling = BlockConvUpsampling(params, [256])
        self.raw2_upsampling = BlockConvUpsampling(params, [256])

    def call(self, inputs, training=True):
        self.endpoints = []

        outputs = self.base_model(inputs, training=training, features_only=True)


        for reduction_idx in self.reduction_indexes:
            endpoint = self.base_model.endpoints['reduction_{}'.format(reduction_idx)]
            self.endpoints.append(endpoint)

        x = self.endpoints[2]

        x = self.raw2_upsampling(x, training=training)
        x = tf.concat([self.endpoints[1], x], -1)

        x = self.raw1_upsampling(x, training=training)
        x = tf.concat([self.endpoints[0], x], -1)

        x = self.raw0_upsampling(x, training=training)

        return self.endpoints, x
