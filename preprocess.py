import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import tensorflow_addons as tfa

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


def distort_color(image, color_ordering=0, fast_mode=True, scope='distort_color'):
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
        saturation_upper = 1.3
        brightness_max_delta = 8. / 255
        hue_max_delta = 0.05
        contrast_lower = 0.9
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

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

def random_expand(image, polys, ratio=2):
    height, width, depth = _ImageDimensions(image, rank=3)

    float_height, float_width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    canvas_width, canvas_height = tf.cast(float_width * ratio, tf.int32), tf.cast(float_height * ratio, tf.int32)

    mean_color_of_image = [0.5, 0.5, 0.5]

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

def rotate_points(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)], axis=0)
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(points, rotation_matrix)

def preprocess_for_train(image, word_poly, image_size, disable_rotation_augmentation):
    if False and tf.random.uniform([], 0, 1) > 0.5:
        image = apply_with_random_selector(image,
                lambda x, ordering: distort_color(x, ordering, True),
                num_cases=4)

    if False and tf.random.uniform([], 0, 1) > 0.5:
        ratio = tf.random.uniform([], minval=1.01, maxval=1.3, dtype=tf.float32)

        poly_rel = word_poly / image_size
        image, new_poly = random_expand(image, poly_rel, ratio)
        image = tf.image.resize(image, [image_size, image_size])
        word_poly = new_poly * image_size

    if tf.random.uniform([], 0, 1) > 0.5:
        angle_min = -20. / 180 * 3.1415
        angle_max = 20. / 180 * 3.1415

        angle = tf.random.uniform([], minval=angle_min, maxval=angle_max, dtype=tf.float32)

        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')

        word_poly -= [image_size/2, image_size/2]
        word_poly = rotate_points(word_poly, angle)
        word_poly += [image_size/2, image_size/2]

    return image, word_poly

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
