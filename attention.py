import logging

import tensorflow as tf

logger = logging.getLogger('detection')

def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32, name='cast_scaled_attention')
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

        self.query_dense = tf.keras.layers.Dense(units=attention_feature_dim)
        self.value_dense = tf.keras.layers.Dense(units=attention_feature_dim)

        self.attention = tf.keras.layers.Attention()

        self.dense = tf.keras.layers.Dense(units=attention_feature_dim)

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

        self.mha = MultiHeadAttention(attention_feature_dim, num_heads)
        self.dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

<<<<<<< HEAD
        self.dense0 = tf.keras.layers.Dense(units=attention_feature_dim)
        self.dense1 = tf.keras.layers.Dense(units=attention_feature_dim)
        self.dense_dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
        self.dense_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
=======
        if False:
            self.dense0 = tf.keras.layers.Dense(units=attention_feature_dim)
            self.dense1 = tf.keras.layers.Dense(units=attention_feature_dim)
            self.dense_dropout = tf.keras.layers.Dropout(rate=params.spatial_dropout)
            self.dense_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
>>>>>>> 8b153855cb79db0e7351ff487026abf698934e14

        self.relu_fn = params.relu_fn

    def call(self, features, state, training):
        x = self.mha(features, state, training)
        x = self.dropout(x, training=training)
        x += features
        attention_out = self.norm(x, training=training)
<<<<<<< HEAD

        x = self.dense0(attention_out)
        x = self.relu_fn(x)
        x = self.dense1(x)
        x = self.dense_dropout(x, training=training)
        x += attention_out
        x = self.dense_norm(x, training=training)
=======
        return attention_out

        if False:
            x = self.dense0(attention_out)
            x = self.relu_fn(x)
            x = self.dense1(x)
            x = self.dense_dropout(x, training=training)
            x += attention_out
            x = self.dense_norm(x, training=training)
>>>>>>> 8b153855cb79db0e7351ff487026abf698934e14

        #logger.info('attention_block: attention_out: {}, x: {}'.format(attention_out.shape, x.shape))

        return x

class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, params, num_units, dictionary_size, cell='lstm', **kwargs):
        super(AttentionCell, self).__init__(self, **kwargs)

        self.dictionary_size = dictionary_size

        cell = cell.lower()
        if cell == 'lstm':
            self.cell = tf.keras.layers.LSTMCell(num_units,
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(),
                    unit_forget_bias=True)
        elif cell == 'gru':
            self.cell = tf.keras.layers.GRUCell(num_units,
                    kernel_initializer=tf.keras.initializers.Orthogonal(),
                    recurrent_initializer=tf.keras.initializers.Orthogonal())
        else:
            raise NameError('unsupported rnn cell type "{}"'.format(cell))

        self.cell_clip = 10.

        attention_feature_dim = 256
        num_heads = 16

        self.dense_features = tf.keras.layers.Dense(units=attention_feature_dim)

        self.attention_layer = AttentionBlock(params, attention_feature_dim, num_heads)
        self.attention_pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.wc = tf.keras.layers.Dense(attention_feature_dim)

        self.wu_pred = tf.keras.layers.Dense(dictionary_size)
        self.wo = tf.keras.layers.Dense(dictionary_size)

    def call(self, char_dist, features, state, training=True):
        state_concat = tf.concat(state, -1)
        state_with_time = tf.expand_dims(state_concat, 1)

        features = self.dense_features(features)

        weighted_features = self.attention_layer(features, state_with_time, training)
        weighted_pooled_features = self.attention_pooling(weighted_features)

        weighted_char_dist = self.wc(char_dist)

        rnn_input = weighted_char_dist + weighted_pooled_features

        #logger.info('char_dist: {}, weighted_char_dist: {}, features: {}, weighted_features: {}, weighted_pooled_features: {}, rnn_input: {}, state: {}'.format(
        #    char_dist.shape, weighted_char_dist.shape, features.shape, weighted_features.shape, weighted_pooled_features.shape, rnn_input.shape, state_with_time.shape))

        rnn_out, new_state = self.cell(rnn_input, state, training=training)
        #new_state = [tf.clip_by_value(s, -self.cell_clip, self.cell_clip) for s in new_state]

        output_char_dist = self.wo(rnn_out) + self.wu_pred(weighted_pooled_features)
        output_char_dist = tf.nn.softmax(output_char_dist, axis=1)

        return output_char_dist, new_state

def add_spatial_encoding(features):
    batch_size = tf.shape(features)[0]
    _, h, w, _ = features.shape.as_list()

    x, y = tf.meshgrid(tf.range(w), tf.range(h))

    w_loc = tf.one_hot(x, w, on_value=1.0, off_value=0.0)
    h_loc = tf.one_hot(y, h, on_value=1.0, off_value=0.0)
    loc = tf.concat([h_loc, w_loc], 2)
    loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])

    return tf.concat([features, loc], 3)

class RNNLayer(tf.keras.layers.Layer):
    def __init__(self, params, num_rnn_units, max_sequence_len, dictionary_size, pad_value, cell='lstm', **kwargs):
        super(RNNLayer, self).__init__(self, **kwargs)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=params.channel_axis,
            momentum=params.batch_norm_momentum,
            epsilon=params.batch_norm_epsilon)

        self.start_token = pad_value
        self.max_sequence_len = max_sequence_len
        self.dictionary_size = dictionary_size

        self.num_rnn_units = num_rnn_units

        self.attention_cell = AttentionCell(params, num_rnn_units, dictionary_size, cell=cell)

    def call(self, image_features, gt_tokens, gt_lens, training):
        image_features = self.bn(image_features, training=training)
        spatial_features = add_spatial_encoding(image_features)

        batch_size = tf.shape(spatial_features)[0]
        spatial_feature_size = spatial_features.shape[-1]
        reshaped_features = tf.reshape(spatial_features, [batch_size, -1, spatial_feature_size])

        #logger.info('image_featuers: {}, spatial features: {} -> {}'.format(image_features.shape, spatial_features.shape, reshaped_features.shape))

        null_token = tf.tile([self.start_token], [batch_size])
        def init():
            state_h = tf.zeros((batch_size, self.num_rnn_units))
            state_c = tf.zeros((batch_size, self.num_rnn_units))
            state = [state_h, state_c]


            char_dists = tf.TensorArray(tf.float32, size=self.max_sequence_len)

            char_dist = tf.one_hot(null_token, self.dictionary_size)

            return state, char_dist, char_dists


        state, char_dist, char_dists = init()
        state_ar, char_dist_ar, char_dists_ar = init()

        for idx in range(self.max_sequence_len):
            if idx != 0:
                if training:
                    char_dist = tf.one_hot(gt_tokens[:, idx-1], self.dictionary_size)

            char_dist, state = self.attention_cell(char_dist, reshaped_features, state, training)
            char_dists = char_dists.write(idx, char_dist)

            char_dist_ar, state_ar = self.attention_cell(char_dist_ar, reshaped_features, state_ar, training)
            char_dists_ar = char_dists_ar.write(idx, char_dist_ar)

        out = char_dists.stack()
        out = tf.transpose(out, [1, 0, 2])

        out_ar = out
        out_ar = char_dists_ar.stack()
        out_ar = tf.transpose(out_ar, [1, 0, 2])

        return out, out_ar
