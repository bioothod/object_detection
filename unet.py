import logging
import re

import tensorflow as tf

logger = logging.getLogger('segmentation')

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
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
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

class Unet(tf.keras.Model):
    def __init__(self, output_channels, base_model):
        super(Unet, self).__init__()

        self.base_model = base_model

        self.up_stack = []
        self.final_endpoints = None

        for name, endpoint in self.base_model.endpoints.items():

        for name, endpoint in self.base_model.endpoints.items():
            if name != 'features':
                m = re.match('reduction_([\d]+)$', name)
                if not m:
                    continue

            endpoint = self.base_model.endpoints[name]

            c = endpoint.shape[3]
            up = upsample(c*2, 3, apply_dropout=True)

            if name == 'features':
                self.up_stack = self.up_stack[:-1]

            self.up_stack.append((name, up))
            logger.info('{}: endpoint: {}, upsampled channels: {}'.format(name, endpoint.shape, c*2))

        self.last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=1, padding='same', activation='softmax')

    def call(self, inputs, training=True):
        x = self.base_model(inputs, training)

        first = True
        for name, up in self.up_stack:
            if not first:
                x = self.base_model.endpoints[name]

            upsampled = up(x)

            if first:
                x = upsampled
            else:
                x = tf.concat([upsampled, endpoint], axis=-1)

            first = True

        x = self.last(x)
        return x
