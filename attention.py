import logging

import tensorflow as tf

import feature_gated_conv as ft
import letters

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

        return outputs, concat_attention

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
        x, s = self.mha(x, state, training)
        attention_out = x + features
        return attention_out, s

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000., (2. * (i // 2)) / tf.cast(d_model, tf.float32, name='cast_pe_0'))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32, name='cast_pe_1')

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, params, attention_feature_dim, num_heads, dictionary_size, **kwargs):
        super().__init__(**kwargs)

        self.relu_fn = params.relu_fn

        self.attention_layer = AttentionBlock(params, attention_feature_dim, num_heads, name='att0')
        self.attention_state_pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.wo = tf.keras.layers.Dense(dictionary_size, name='wo')

    def call(self, features, training):
        weighted_features, att_state = self.attention_layer(features, features, training)

        output_char_dist = self.wo(weighted_features)
        output_char_dist = tf.nn.softmax(output_char_dist, axis=1)

        return output_char_dist

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
        super().__init__(**kwargs)

        self.start_token = pad_value
        self.max_sequence_len = max_sequence_len
        self.dictionary_size = dictionary_size

        self.attention_feature_dim = 128
        num_heads = 4

        self.res0 = ft.GatedBlockResidual(params, [256, 256], kernel_size2=(3, 6), name='res0')
        self.pool0 = ft.BlockPool(params, 128, strides=(2, 1), name='pool0')
        self.res1 = ft.GatedBlockResidual(params, [128, 128], kernel_size2=(2, 4), name='res1')
        self.pool1 = ft.BlockPool(params, 128, strides=(2, 1), name='pool1')
        self.res2 = ft.GatedBlockResidual(params, [128, 128], kernel_size2=(2, 4), name='res2')
        self.pool2 = ft.BlockPool(params, 128, strides=(2, 1), name='pool2')

        self.positional_encoding = PositionalEncoding(params.crop_size[1], self.attention_feature_dim)

        #self.attention_conv = ft.TextConv(params, self.attention_feature_dim, kernel_size=(1, 1), strides=(1, 1))
        #self.attention_conv = letters.LettersConv(params, self.attention_feature_dim, kernel_size=1, strides=1)

        self.attention_cell = AttentionCell(params, self.attention_feature_dim, num_heads, dictionary_size)

    def call(self, image_features, gt_tokens, gt_lens, training):
        batch_size = tf.shape(image_features)[0]

        img = self.res0(image_features, training)
        img = self.pool0(img, training)
        img = self.res1(img, training)
        img = self.pool1(img, training)
        img = self.res2(img, training)
        img = self.pool2(img, training)

        #pos_features = add_spatial_encoding(img)

        reshaped_img = tf.reshape(img, [batch_size, -1, tf.shape(img)[-1]])
        features = self.positional_encoding(reshaped_img)
        #features = self.attention_conv(features, training)


        #logger.info('image_features: {}, img: {}, spatial features: {}'.format(image_features.shape, img.shape, spatial_features.shape))

        #features = tf.reshape(conv_features, [batch_size, -1, self.attention_feature_dim])

        char_dists = self.attention_cell(features, training)

        return char_dists, char_dists
