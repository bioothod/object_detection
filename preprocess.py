import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import tensorflow_addons as tfa
import autoaugment

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
        saturation_lower = 0.7
        saturation_upper = 1.5
        brightness_max_delta = 16. / 255
        hue_max_delta = 0.2
        contrast_lower = 0.7
        contrast_upper = 1.5
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

        return image

def random_expand(image, polys, ratio):
    height, width, depth = _ImageDimensions(image, rank=3)

    float_height, float_width = tf.cast(height, ratio.dtype), tf.cast(width, ratio.dtype)

    canvas_width, canvas_height = tf.cast(float_width * ratio, tf.int32), tf.cast(float_height * ratio, tf.int32)

    mean_color_of_image = [128, 128, 128]

    x = tf.random.uniform([], minval=0, maxval=canvas_width - width, dtype=tf.int32)
    y = tf.random.uniform([], minval=0, maxval=canvas_height - height, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[y, canvas_height - height - y], [x, canvas_width - width - x]])

    big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values = mean_color_of_image[0]),
                          tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values = mean_color_of_image[1]),
                          tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values = mean_color_of_image[2])], axis=-1)

    scale = tf.cast(tf.stack([width, height]), polys.dtype)
    offset = tf.cast(tf.stack([x, y]), polys.dtype)
    absolute_polys = polys * scale + offset 

    new_scale = tf.cast(tf.stack([canvas_width, canvas_height]), polys.dtype)
    polys = absolute_polys / new_scale

    return big_canvas, polys

@tf.function
def random_crop(image, polys, text_labels):
    height, width, depth = _ImageDimensions(image, rank=3)
    dtype = image.dtype

    float_height, float_width = tf.cast(height, polys.dtype), tf.cast(width, polys.dtype)

    for i in tf.range(20):
        crop_x0 = tf.random.uniform([], minval=0, maxval=0.7) * float_width
        crop_y0 = tf.random.uniform([], minval=0, maxval=0.7) * float_height

        crop_width_max = float_width - crop_x0
        crop_x1 = tf.random.uniform([], minval=crop_width_max/2, maxval=crop_width_max) + crop_x0
        
        crop_height_max = float_height - crop_y0
        crop_y1 = tf.random.uniform([], minval=crop_height_max/2, maxval=crop_height_max) + crop_y0

        x = polys[..., 0]
        y = polys[..., 1]

        xmin = tf.reduce_min(x, 1)
        xmax = tf.reduce_max(x, 1)
        ymin = tf.reduce_min(y, 1)
        ymax = tf.reduce_max(y, 1)

        has_x0 = tf.math.less_equal(crop_x0, xmin)
        has_x1 = tf.math.greater_equal(crop_x1, xmax)
        has_y0 = tf.math.less_equal(crop_y0, ymin)
        has_y1 = tf.math.greater_equal(crop_y1, ymax)

        overlap_mask = tf.logical_and(tf.logical_and(has_x0, has_x1), tf.logical_and(has_y0, has_y1))

        if tf.math.count_nonzero(overlap_mask) != 0:
            polys = tf.boolean_mask(polys, overlap_mask)

            x = polys[..., 0]
            y = polys[..., 1]

            x = (x - crop_x0) / (crop_x1 - crop_x0) * float_width
            y = (y - crop_y0) / (crop_y1 - crop_y0) * float_height
            polys = tf.stack([x, y], -1)

            text_labels = tf.boolean_mask(text_labels, overlap_mask)

            crop_x0 = tf.cast(crop_x0, tf.int32)
            crop_x1 = tf.cast(crop_x1, tf.int32)
            crop_y0 = tf.cast(crop_y0, tf.int32)
            crop_y1 = tf.cast(crop_y1, tf.int32)

            image = tf.image.crop_to_bounding_box(image, crop_y0, crop_x0, crop_y1 - crop_y0, crop_x1 - crop_x0)
            image = tf.image.resize(image, [height, width])
            image = tf.cast(image, dtype)
            break

    return image, polys, text_labels

def rotate_points(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)], axis=0)
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(points, rotation_matrix)

def preprocess_for_train(image, word_poly, text_labels, image_size, disable_rotation_augmentation, use_random_augmentation):
    dtype = image.dtype

    resize_rnd = tf.random.uniform([], 0, 1)
    if resize_rnd > 0.3:
        if resize_rnd > 0.6:
            ratio = tf.random.uniform([], minval=1.1, maxval=2., dtype=word_poly.dtype)

            current_image_size = tf.cast(tf.shape(image)[1], word_poly.dtype)
            poly_rel = word_poly / current_image_size
            image, new_poly = random_expand(image, poly_rel, ratio)
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.cast(image, dtype)
            word_poly = new_poly * image_size
        else:
            image, word_poly, text_labels = random_crop(image, word_poly, text_labels)

    if tf.random.uniform([], 0, 1) > 0.5:
        if use_random_augmentation:
            randaug_num_layers = 1
            randaug_magnitude = 11

            image = autoaugment.distort_image_with_randaugment(image, randaug_num_layers, randaug_magnitude)
            image = tf.cast(image, dtype)
        else:
            image = apply_with_random_selector(image,
                    lambda x, ordering: distort_color(x, ordering, fast_mode=False),
                    num_cases=4)
            image = tf.cast(image, dtype)

    if not disable_rotation_augmentation and tf.random.uniform([], 0, 1) > 0.5:
        angle_min = -5. / 180 * 3.1415
        angle_max = 5. / 180 * 3.1415

        angle = tf.random.uniform([], minval=angle_min, maxval=angle_max, dtype=tf.float32)

        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')

        angle = tf.cast(angle, word_poly.dtype)
        current_image_size = tf.cast(tf.shape(image)[1], word_poly.dtype)
        word_poly -= [current_image_size/2, current_image_size/2]
        word_poly = rotate_points(word_poly, angle)
        word_poly += [current_image_size/2, current_image_size/2]

    return image, word_poly, text_labels

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
