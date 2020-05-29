import logging
import typing

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import tensorflow_addons as tfa
import autoaugment

logger = logging.getLogger('detection')

def _ImageDimensions(image, rank = 3):
    """Returns the dimensions of an image tensor.

    Args:
      image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
      rank: The expected rank of the image

    Returns:
      A list of corresponding to the dimensions of the
      input image.  Dimensions that are statically known are python integers,
      otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=False, scope='distort_color'):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """

    with tf.name_scope(scope):
        saturation_lower = 0.9
        saturation_upper = 1.2
        brightness_max_delta = 8 / 255
        hue_max_delta = 0.2
        contrast_lower = 0.9
        contrast_upper = 1.2
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
            else:
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=hue_max_delta)
                image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
                image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
                image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        return tf.clip_by_value(image, 0, 1)

def random_expand(image, polys, ratio):
    height, width, depth = _ImageDimensions(image, rank=3)

    float_height = tf.cast(height, ratio.dtype)
    float_width = tf.cast(width, ratio.dtype)

    canvas_width = tf.cast(float_width * ratio, tf.int32)
    canvas_height = tf.cast(float_height * ratio, tf.int32)

    canvas_size = tf.maximum(canvas_height, canvas_width)

    mean_color_of_image = [128., 128., 128.]

    if canvas_size == width:
        x = 0
    else:
        x = tf.random.uniform([], minval=0, maxval=canvas_size-width, dtype=tf.int32)

    if canvas_size == height:
        y = 0
    else:
        y = tf.random.uniform([], minval=0, maxval=canvas_size-height, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[y, canvas_size-height-y], [x, canvas_size-width-x]])
    #tf.print('float_width:', float_width, ', float_height:', float_height, ', canvas_width:', canvas_width, ', canvas_height:', canvas_height, ', canvas_size:', canvas_size, ', paddings:', paddings)

    big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values=mean_color_of_image[0]),
                           tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values=mean_color_of_image[1]),
                           tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values=mean_color_of_image[2])],
                    axis=-1)

    canvas_offset = tf.cast(tf.stack([x, y]), polys.dtype)
    polys += canvas_offset

    return big_canvas, polys

@tf.function
def random_crop(image: tf.Tensor, word_poly: tf.Tensor, labels: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dtype = image.dtype

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_height_float = tf.cast(image_height, word_poly.dtype)
    image_width_float = tf.cast(image_width, word_poly.dtype)

    # Get random crop dims
    crop_factor = tf.linspace(.4, .8, 4)
    idx = tf.random.uniform([1], maxval=4, dtype=tf.int32)
    crop_factor = tf.gather(crop_factor, idx)
    crop_width = tf.cast(image_width_float * crop_factor, tf.int32)
    crop_height = tf.cast(image_height_float * crop_factor, tf.int32)

    # Pick coordinates to start the crop
    crop_x = tf.random.uniform(shape=[1], maxval=(image_width - crop_width)[0], dtype=tf.int32)
    crop_y = tf.random.uniform(shape=[1], maxval=(image_height - crop_height)[0], dtype=tf.int32)

    # Crop the image and resize it back to original size
    crop_image = tf.image.crop_to_bounding_box(image, crop_y[0], crop_x[0], crop_height[0], crop_width[0])
    crop_image = tf.image.resize(crop_image, [image_height, image_width])
    crop_image = tf.cast(crop_image, dtype)

    # Cast crop coordinates to float, so they can be used for clipping
    crop_x = tf.cast(crop_x, word_poly.dtype)
    crop_width = tf.cast(crop_width, word_poly.dtype)
    crop_y = tf.cast(crop_y, word_poly.dtype)
    crop_height = tf.cast(crop_height, word_poly.dtype)

    px = word_poly[..., 0]
    py = word_poly[..., 1]

    px -= crop_x
    py -= crop_y

    px = tf.clip_by_value(px, 0, crop_width)
    py = tf.clip_by_value(py, 0, crop_height)

    px0 = tf.math.reduce_min(px, 1)
    py0 = tf.math.reduce_min(py, 1)
    px1 = tf.math.reduce_max(px, 1)
    py1 = tf.math.reduce_max(py, 1)

    widths = px1 - px0
    heights = py1 - py0
    areas = widths * heights

    # Min area is the 1 per cent of the whole area
    min_area = 0.01 * (crop_height * crop_height)
    large_areas = tf.reshape(tf.math.greater_equal(areas, min_area), [-1])

    # Get only large enough polygons
    px = tf.boolean_mask(px, large_areas, axis=0)
    py = tf.boolean_mask(py, large_areas, axis=0)
    labels = tf.boolean_mask(labels, large_areas)

    # Scale the boxes to original image
    ratio_w = image_width_float / crop_width
    ratio_h = image_height_float / crop_height

    px *= ratio_w
    py *= ratio_h

    word_poly = tf.stack([px, py], -1)

    return crop_image, word_poly, labels

def rotate_points(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)], axis=0)
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(points, rotation_matrix)

def preprocess_for_train(image, word_poly, text_labels, rotation_augmentation, use_augmentation, dtype):
    # image is a squared dtype already

    if tf.random.uniform([], 0, 1) > 0.5:
        x = word_poly[..., 0]
        y = word_poly[..., 1]
        max_ratio = 1.3

        min_size = 16

        dx = tf.cast(tf.shape(image)[1], tf.float32)
        dy = tf.cast(tf.shape(image)[0], tf.float32)

        #tf.print('dx:', dx, ', dy:', dy, ', x:', x, ', y:', y)

        dx = tf.minimum(dx, tf.abs(x[..., 0] - x[..., 1]))
        dx = tf.minimum(dx, tf.abs(x[..., 1] - x[..., 2]))
        dx = tf.minimum(dx, tf.abs(x[..., 2] - x[..., 3]))
        dx = tf.minimum(dx, tf.abs(x[..., 3] - x[..., 0]))

        dy = tf.minimum(dy, tf.abs(y[..., 0] - y[..., 1]))
        dy = tf.minimum(dy, tf.abs(y[..., 1] - y[..., 2]))
        dy = tf.minimum(dy, tf.abs(y[..., 2] - y[..., 3]))
        dy = tf.minimum(dy, tf.abs(y[..., 3] - y[..., 0]))

        dx = tf.reduce_min(dx)
        dy = tf.reduce_min(dy)

        min_dist = tf.minimum(dx, dy)

        if min_dist >= max_ratio * min_size and tf.random.uniform([], 0, 1) > 0.5 and tf.shape(x)[0] > 0:
            maxval = min_dist/min_size
            ratio = tf.random.uniform([], minval=1.01, maxval=maxval, dtype=word_poly.dtype)

            #tf.print('dx:', dx, ', dy:', dy, ', min_dist:', min_dist, ', min_size:', min_size, ', maxval:', maxval, ', ratio:', ratio)
            image, word_poly = random_expand(image, word_poly, ratio)
        else:
            image, word_poly, text_labels = random_crop(image, word_poly, text_labels)

    if use_augmentation and tf.random.uniform([], 0, 1) > 0.5:
        for aug in use_augmentation.split(','):
            if aug == 'speckle':
                image = image + image * tf.random.normal(tf.shape(image), mean=0, stddev=0.1)
                image = tf.clip_by_value(image, 0, 255)
            elif aug == 'v0':
                image = tf.cast(image, tf.uint8)
                image = autoaugment.distort_image_with_autoaugment(image, 'v0')
                image = tf.cast(image, dtype)
            elif aug == 'random':
                randaug_num_layers = 1
                randaug_magnitude = 11

                image = tf.cast(image, tf.uint8)
                image = autoaugment.distort_image_with_randaugment(image, randaug_num_layers, randaug_magnitude)
                image = tf.cast(image, dtype)
            elif 'color' in aug:
                # image must be in [0, 1] range for this function
                image /= 255

                if aug == 'color_fast_mode':
                    fast_mode = True
                    num_cases = 2
                else:
                    fast_mode = False
                    num_cases = 4

                image = apply_with_random_selector(image,
                        lambda x, ordering: distort_color(x, ordering, fast_mode=fast_mode),
                        num_cases=num_cases)

                image *= 255

    if rotation_augmentation > 0 and tf.random.uniform([], 0, 1) > 0.5:
        angle_min = -float(rotation_augmentation) / 180 * 3.1415
        angle_max = float(rotation_augmentation) / 180 * 3.1415

        angle = tf.random.uniform([], minval=angle_min, maxval=angle_max, dtype=tf.float32)

        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')

        angle = tf.cast(angle, word_poly.dtype)
        current_image_size = tf.cast(tf.shape(image)[1], word_poly.dtype)
        word_poly -= [current_image_size/2, current_image_size/2]
        word_poly = rotate_points(word_poly, angle)
        word_poly += [current_image_size/2, current_image_size/2]

    image = tf.cast(image, dtype)
    image -= 128
    image /= 128

    return image, word_poly, text_labels

def preprocess_for_evaluation(image, dtype):
    # image is a squared dtype already

    image -= 128
    image /= 128
    return image

def pad_resize_image(image, dims):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    mx = tf.maximum(h, w)

    dy = mx - h
    dx = mx - w

    dy0 = tf.cast(dy / 2, tf.int32)
    dx0 = tf.cast(dx / 2, tf.int32)

    image = tf.pad(image, [[dy0, dy - dy0], [dx0, dx - dx0], [0, 0]])
    image = tf.image.resize(image, dims, preserve_aspect_ratio=True)

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[sy, syd - sy], [sx, sxd - sx], [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=128)
    return image
