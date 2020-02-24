import logging

import tensorflow as tf

import feature_gated_conv as ft

logger = logging.getLogger('detection')

def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], value.dtype, name='cast_scaled_attention')
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_feature_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_feature_dim = attention_feature_dim

        assert attention_feature_dim % self.num_heads == 0

        self.depth = attention_feature_dim // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=attention_feature_dim, name='query')
        self.value_dense = tf.keras.layers.Dense(units=attention_feature_dim, name='value')

        #self.attention = tf.keras.layers.Attention()

        self.dense = tf.keras.layers.Dense(units=attention_feature_dim, name='final')

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, features, state_with_time, training):
        batch_size = tf.shape(features)[0]

        # linear layers
        query = self.query_dense(state_with_time)
        value = self.value_dense(features)

        # split heads
        query = self.split_heads(query, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, value, value, mask=None)
        #scaled_attention = self.attention([query, value])

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.attention_feature_dim))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, params, attention_feature_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        if params.dtype == tf.float32:
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.norm = tf.keras.layers.BatchNormalization(
                axis=params.channel_axis,
                momentum=params.batch_norm_momentum,
                epsilon=params.batch_norm_epsilon)

        self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
        self.mha = MultiHeadAttention(attention_feature_dim, num_heads, name='mha')

        self.dense0 = tf.keras.layers.Dense(units=attention_feature_dim, name='dense0')
        self.dense1 = tf.keras.layers.Dense(units=attention_feature_dim, name='dense1')
        self.dense_dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
        if params.dtype == tf.float32:
            self.dense_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.dense_norm = tf.keras.layers.BatchNormalization(
                axis=params.channel_axis,
                momentum=params.batch_norm_momentum,
                epsilon=params.batch_norm_epsilon)

        self.relu_fn = params.relu_fn

    def call(self, features, state, training):
        x = self.norm(features, training=training)
        x = self.dropout(x, training=training)
        x = self.mha(x, state, training)
        attention_out = x + features
        return attention_out

        x = self.dense_norm(x, training=training)
        x = self.dense_dropout(x, training=training)

        x = self.dense0(x)
        x = self.relu_fn(x)
        x = self.dense1(x)
        x = self.relu_fn(x)

        dense_out = x + attention_out

        #logger.info('attention_block: attention_out: {}, x: {}'.format(attention_out.shape, x.shape))

        return dense_out

class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, params, attention_feature_dim, num_heads, dictionary_size, **kwargs):
        super(AttentionCell, self).__init__(self, **kwargs)

        self.attention_layer = AttentionBlock(params, attention_feature_dim, num_heads, name='att0')
        self.attention_pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.wc = tf.keras.layers.Dense(attention_feature_dim, name='wc')

        self.dense_dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
        self.dense0 = tf.keras.layers.Dense(attention_feature_dim, name='dense0')

        self.wo = tf.keras.layers.Dense(dictionary_size, name='wo')

    def call(self, char_dist, features, state, training):
        state_with_time = tf.expand_dims(state, 1)

        weighted_features = self.attention_layer(features, state_with_time, training)
        weighted_pooled_features = self.attention_pooling(weighted_features)

        weighted_char_dist = self.wc(char_dist)

        net_input = weighted_char_dist + weighted_pooled_features

        net_out = self.dense_dropout(net_input, training=training)
        net_out = self.dense0(net_out)

        output_char_dist = self.wo(net_out)
        output_char_dist = tf.nn.softmax(output_char_dist, axis=1)

        return output_char_dist, weighted_pooled_features

def add_spatial_encoding(features):
    batch_size = tf.shape(features)[0]
    h = tf.shape(features)[1]
    w = tf.shape(features)[2]

    x, y = tf.meshgrid(tf.range(w), tf.range(h))

    w_loc = tf.one_hot(x, w, dtype=features.dtype)
    h_loc = tf.one_hot(y, h, dtype=features.dtype)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])

    return tf.concat([features, loc], 3)

class RNNLayer(tf.keras.Model):
    def __init__(self, params, max_sequence_len, dictionary_size, pad_value, **kwargs):
        super(RNNLayer, self).__init__(self, **kwargs)

        self.start_token = pad_value
        self.max_sequence_len = max_sequence_len
        self.dictionary_size = dictionary_size

        self.attention_feature_dim = 256
        num_heads = 8

        self.res0 = ft.GatedBlockResidual(params, [128, 256], name='res0')
        self.pool0 = ft.BlockPool(params, 128, strides=(2, 1), name='pool0')
        self.res1 = ft.GatedBlockResidual(params, [64, 128], name='res1')
        self.pool1 = ft.BlockPool(params, 128, strides=(2, 1), name='pool1')
        self.res2 = ft.GatedBlockResidual(params, [64, 128], name='res2')
        self.pool2 = ft.BlockPool(params, 128, strides=(2, 1), name='pool2')
        self.attention_conv = ft.TextConv(params, self.attention_feature_dim, kernel_size=1, strides=1)

        self.attention_cell = AttentionCell(params, self.attention_feature_dim, num_heads, dictionary_size)

    def call(self, image_features, gt_tokens, gt_lens, training):
        #img = self.stem(image_features, training)
        img = self.res0(image_features, training)
        img = self.pool0(img, training)
        img = self.res1(img, training)
        img = self.pool1(img, training)
        img = self.res2(img, training)
        img = self.pool2(img, training)

        spatial_features = add_spatial_encoding(img)

        spatial_features = self.attention_conv(spatial_features, training)

        batch_size = tf.shape(spatial_features)[0]

        #logger.info('image_features: {}, img: {}, spatial features: {}'.format(image_features.shape, img.shape, spatial_features.shape))

        features = tf.reshape(spatial_features, [batch_size, -1, self.attention_feature_dim])

        null_token = tf.tile([self.start_token], [batch_size])
        def init():
            char_dists = tf.TensorArray(image_features.dtype, size=self.max_sequence_len)

            char_dist = tf.one_hot(null_token, self.dictionary_size, dtype=image_features.dtype)
            initial_state = tf.zeros((batch_size, tf.shape(features)[-1]), dtype=features.dtype)

            return initial_state, char_dist, char_dists


        state, char_dist, char_dists = init()
        state_ar, char_dist_ar, char_dists_ar = init()

        for idx in range(self.max_sequence_len):
            if idx != 0:
                if training:
                    char_dist = tf.one_hot(gt_tokens[:, idx-1], self.dictionary_size, dtype=image_features.dtype)

            if training:
                char_dist, state = self.attention_cell(char_dist, features, state, training)
                char_dists = char_dists.write(idx, char_dist)

            char_dist_ar, state_ar = self.attention_cell(char_dist_ar, features, state_ar, training)
            char_dists_ar = char_dists_ar.write(idx, char_dist_ar)

        out_ar = char_dists_ar.stack()
        out_ar = tf.transpose(out_ar, [1, 0, 2])

        if training:
            out = char_dists.stack()
            out = tf.transpose(out, [1, 0, 2])
        else:
            out = out_ar

        return out, out_ar
