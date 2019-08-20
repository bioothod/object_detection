import logging
import re

import tensorflow as tf

logger = logging.getLogger('segmentation')

def upsample(filters, size, norm_type='batchnorm', apply_dropout=True):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.ReLU())

  return result

class Unet(tf.keras.Model):
    def __init__(self, output_channels, down_stack):
        super(Unet, self).__init__()

        self.down_stack = down_stack
        self.up_stack = [
            upsample(512, 3),  # 4x4 -> 8x8
            upsample(256, 3),  # 8x8 -> 16x16
            upsample(128, 3),  # 16x16 -> 32x32
            upsample(64, 3),   # 32x32 -> 64x64
        ]

        self.last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')

    def call(self, inputs, training=True):
        skips = self.down_stack(inputs, training)

        x = skips[-1]
        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            upsampled = up(x)
            logger.info('x: {}, up: {}, skip: {}'.format(x.shape, upsampled.shape, skip.shape))

            x = tf.concat([upsampled, skip], axis=-1)

        ret = self.last(x)
        logger.info('output: x: {}, ret: {}'.format(x.shape, ret.shape))
        return ret
