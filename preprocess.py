import logging

import tensorflow as tf

import autoaugment

logger = logging.getLogger('vggface_emotions')

def pad_resize_image(image, dims):
    image = tf.image.resize(image, dims, preserve_aspect_ratio=True)

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[sy, syd - sy], [sx, sxd - sx], [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=128)
    image = tf.cast(image, dtype=tf.uint8)
    return image

# do not use padded resize, since bbox scaling does not yet support this
def simple_resize_image(image, dims):
    image = tf.image.resize(image, dims)
    image = tf.cast(image, dtype=tf.uint8)
    return image

def try_resize(image, output_height, output_width):
    shape = tf.shape(image)
    image = tf.cond(tf.logical_and(tf.equal(shape[0], output_height),
                                   tf.equal(shape[1], output_width)),
                    lambda: image,
                    #lambda: pad_resize_image(image, [output_height, output_width]))
                    lambda: simple_resize_image(image, [output_height, output_width]))
    return image

def mean_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]

    return tf.concat(axis=2, values=channels)

_vgg_means = [91.4953, 103.8827, 131.0912]

def prepare_image_for_training(image, output_height, output_width, dtype=tf.float32, autoaugment_name=None):
    max_scale = 1
    image = try_resize(image, int(max_scale*output_height), int(max_scale*output_width))
    #image = tf.image.random_flip_left_right(image)

    if autoaugment_name:
        image = autoaugment.distort_image_with_autoaugment(image, autoaugment_name)

    image = tf.image.convert_image_dtype(image, dtype=dtype)

    return image

def prepare_image_for_evaluation(image, output_height, output_width, dtype=tf.float32):
    image = try_resize(image, output_height, output_width)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image

def processing_function(image, output_height, output_width, is_training, dtype=tf.float32, autoaugment_name=None):
    if is_training:
        return prepare_image_for_training(image, output_height, output_width, dtype, autoaugment_name)
    else:
        return prepare_image_for_evaluation(image, output_height, output_width, dtype)
