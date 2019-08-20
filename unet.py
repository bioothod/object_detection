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
    def __init__(self, output_channels, base_model):
        super(Unet, self).__init__()

        self.up_stack = []

        layers = ('top_activation', 'block6a_expand_activation', 'block4a_expand_activation', 'block3a_expand_activation', 'block2a_expand_activation')
        layers = [base_model.get_layer(name).output for name in layers]

        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        decoder_filters=(512, 256, 128, 64, 32, 16)
        for l, f in zip(layers, decoder_filters):
            up = upsample(f, 3, apply_dropout=True)
            self.up_stack.append(up)

        self.last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')

    def call(self, inputs, training=True):
        skips = self.down_stack(inputs, training)

        for s in skips:
            logger.info('skips: {}: {}'.format(s.name, s.shape))

        logger.info('down_stack: trainable vars: {}'.format(len(self.down_stack.trainable_variables)))

        x = skips[0]
        skips = (skips[1:])

        for up, skip in zip(self.up_stack, skips):
            upsampled = up(x)

            logger.info('inputs: {}, x: {}, upsampled: {}, skip: {}'.format(inputs.shape, x.shape, upsampled.shape, skip.shape))
            x = tf.concat([upsampled, skip], axis=-1)

        ret = self.last(x)
        return ret
