import typing

import numpy as np
import tensorflow as tf

def num_params_flops(variables, readable_format=True):
  """Return number of parameters and flops."""
  nparams = np.sum([np.prod(v.get_shape().as_list()) for v in variables])
  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(tf.get_default_graph(), options=options).total_float_ops
  if readable_format:
    nparams = float(nparams)  * 1e-6
    flops = float(flops) * 1e-9
  return nparams, flops

class Resize(tf.keras.layers.Layer):
    def __init__(self, features: int, **kwargs):
        super().__init__()
        self.antialiasing_conv = ConvBlock(features,
                                           separable=True,
                                           kernel_size=3,
                                           padding='same')

    def call(self,
             images: tf.Tensor,
             target_dim: typing.Tuple[int, int, int, int] = None,
             training: bool = True) -> tf.Tensor:
        dims = target_dim[1:3]
        x = tf.image.resize(images, dims, method='nearest')
        x = self.antialiasing_conv(x, training=training)
        return x

class ConvBlock(tf.keras.layers.Layer):

    def __init__(self,
                 features: int = None,
                 separable: bool = False,
                 activation: str = None,
                 **kwargs):
        super().__init__()

        if separable:
            self.conv = tf.keras.layers.SeparableConv2D(filters=features, **kwargs)
        else:
            self.conv = tf.keras.layers.Conv2D(features, **kwargs)
        self.bn = tf.keras.layers.BatchNormalization()

        if activation == 'swish':
            self.activation = tf.keras.layers.Activation(tf.nn.swish)
        elif activation is not None:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = tf.keras.layers.Activation('linear')

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.bn(x, training=training)
        x = self.conv(x)
        x = self.activation(x)
        return x

EPSILON = 1e-8

class FastFusion(tf.keras.layers.Layer):
    def __init__(self, size: int, features: int, dtype: tf.dtypes.DType = tf.float32):
        super(FastFusion, self).__init__()

        self.size = size
        w_init = tf.keras.initializers.constant(1. / size)
        self.w = tf.Variable(name='w',
                             initial_value=tf.cast(w_init(shape=(size,)), dtype=dtype),
                             dtype=dtype,
                             trainable=True)
        self.relu = tf.keras.layers.Activation('relu')

        self.conv = ConvBlock(features,
                              separable=True,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              activation='swish')

        self.resize = Resize(features)

    def call(self,
             inputs: typing.Sequence[tf.Tensor],
             training: bool = True) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs: List[tf.Tensor] of shape (BATCH, H, W, C)
        """

        # wi has to be larger than 0 -> Apply ReLU
        w = self.relu(self.w)
        w_sum = EPSILON + tf.reduce_sum(w, axis=0)

        # [INPUTS, BATCH, H, W, C]
        # The last feature map has to be resized according to the
        # other inputs

        weighted_inputs = []
        for i in range(self.size):
            if i == self.size - 1:
                inp = self.resize(inputs[-1], tf.shape(inputs[0]), training=training)
            else:
                inp = inputs[i]

            weighted_inputs.append(w[i] * inp)

        # Sum weighted inputs
        # (BATCH, H, W, C)
        weighted_sum = tf.reduce_sum(weighted_inputs, axis=0) / w_sum
        return self.conv(weighted_sum, training=training)

class BiFPNBlock(tf.keras.Model):

    def __init__(self, features: int, dtype: tf.dtypes.DType = tf.float32, **kwargs: dict):
        super().__init__(**kwargs)

        # Feature fusion for intermediate level
        # ff stands for Feature fusion
        # td refers to intermediate level
        self.ff_6_td = FastFusion(2, features, dtype=dtype)
        self.ff_5_td = FastFusion(2, features, dtype=dtype)
        self.ff_4_td = FastFusion(2, features, dtype=dtype)

        # Feature fusion for output
        self.ff_7_out = FastFusion(2, features, dtype=dtype)
        self.ff_6_out = FastFusion(3, features, dtype=dtype)
        self.ff_5_out = FastFusion(3, features, dtype=dtype)
        self.ff_4_out = FastFusion(3, features, dtype=dtype)
        self.ff_3_out = FastFusion(2, features, dtype=dtype)

    def call(self,
             features: typing.Sequence[tf.Tensor],
             training: bool = True) -> typing.Sequence[tf.Tensor]:
        """
        Computes the feature fusion of bottom-up features comming
        from the Backbone NN

        Parameters
        ----------
        features: List[tf.Tensor]
            Feature maps of each convolutional stage of the
            backbone neural network
        """
        P3, P4, P5, P6, P7 = features

        # Compute the intermediate state
        # Note that P3 and P7 have no intermediate state
        P6_td = self.ff_6_td([P6, P7], training=training)
        P5_td = self.ff_5_td([P5, P6_td], training=training)
        P4_td = self.ff_4_td([P4, P5_td], training=training)

        # Compute out features maps
        P3_out = self.ff_3_out([P3, P4_td], training=training)
        P4_out = self.ff_4_out([P4, P4_td, P3_out], training=training)
        P5_out = self.ff_5_out([P5, P5_td, P4_out], training=training)
        P6_out = self.ff_6_out([P6, P6_td, P5_out], training=training)
        P7_out = self.ff_7_out([P7, P6_td], training=training)

        return [P3_out, P4_out, P5_out, P6_out, P7_out]


class BiFPN(tf.keras.Model):

    def __init__(self, features: int = 64, n_blocks: int = 3, dtype: tf.dtypes.DType = tf.float32, **kwargs: dict):
        super(BiFPN, self).__init__(**kwargs)

        # One pixel-wise for each feature comming from the
        # bottom-up path
        self.pixel_wise = [ConvBlock(features, kernel_size=1) for _ in range(3)]

        self.gen_P6 = ConvBlock(features,
                                kernel_size=3,
                                strides=2,
                                padding='same')

        self.relu = tf.keras.layers.Activation('relu')

        self.gen_P7 = ConvBlock(features,
                                kernel_size=3,
                                strides=2,
                                padding='same')

        self.blocks = [BiFPNBlock(features, dtype=dtype) for i in range(n_blocks)]

    def call(self,
             inputs: typing.Sequence[tf.Tensor],
             training: bool = True) -> typing.Sequence[tf.Tensor]:

        # Each Pin has shape (BATCH, H, W, C)
        # We first reduce the channels using a pixel-wise conv
        _, _, *C = inputs
        P3, P4, P5 = [self.pixel_wise[i](C[i], training=training) for i in range(len(C))]
        P6 = self.gen_P6(C[-1], training=training)
        P7 = self.gen_P7(self.relu(P6), training=training)

        x = [P3, P4, P5, P6, P7]

        for block in self.blocks:
            x = block(x, training=training)
        return x

