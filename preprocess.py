import logging

import tensorflow as tf

import autoaugment

logger = logging.getLogger('detection')

def pad_resize_image(image, dims, fill_constant=0):
    orig_dtype = image.dtype

    image = tf.image.resize(image, dims, preserve_aspect_ratio=True)

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[sy, syd - sy], [sx, sxd - sx], [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=fill_constant)
    image = tf.cast(image, dtype=orig_dtype)
    return image

# do not use padded resize, since bbox scaling does not yet support this
def simple_resize_image(image, dims):
    image = tf.image.resize(image, dims)
    return image

def try_resize(image, output_height, output_width):
    shape = tf.shape(image)
    image = tf.cond(tf.logical_and(tf.equal(shape[0], output_height),
                                   tf.equal(shape[1], output_width)),
                    lambda: image,
                    lambda: pad_resize_image(image, [output_height, output_width]))
                    #lambda: simple_resize_image(image, [output_height, output_width]))
    return image

def normalize(image, dtype):
    image = tf.cast(image, dtype)
    #vgg_means = tf.constant([91.4953, 103.8827, 131.0912])
    #image -= vgg_means
    image -= 128.
    image /= 128.

    return image

def prepare_image_for_training(image, output_height, output_width, dtype=tf.float32, autoaugment_name=None):
    max_scale = 1
    image = try_resize(image, int(max_scale*output_height), int(max_scale*output_width))

    if autoaugment_name:
        image = autoaugment.distort_image_with_autoaugment(image, autoaugment_name)

    image = normalize(image, dtype)

    return image

def prepare_image_for_evaluation(image, output_height, output_width, dtype=tf.float32):
    image = try_resize(image, output_height, output_width)
    image = normalize(image, dtype)
    return image

def processing_function(image, output_height, output_width, is_training, dtype=tf.float32, autoaugment_name=None):
    if is_training:
        return prepare_image_for_training(image, output_height, output_width, dtype, autoaugment_name)
    else:
        return prepare_image_for_evaluation(image, output_height, output_width, dtype)
