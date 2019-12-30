import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

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
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
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

def preprocess_for_train(image, char_poly, word_poly, disable_rotation_augmentation):
    image = apply_with_random_selector(image,
            lambda x, ordering: distort_color(x, ordering, True),
            num_cases=4)

    char_poly_rel = char_poly / image_size
    word_poly_rel = word_poly / image_size
    poly_rel = tf.concat([char_poly_rel, word_poly_rel], axis=0)

    ratio = tf.random.uniform([], minval=1.2, maxval=2.5, dtype=tf.float32)

    image, new_poly = random_expand(image, poly_rel, ratio)
    image = tf.image.resize(image, [image_size, image_size])

    new_poly *= image_size

    size_splits = [tf.shape(char_poly)[0], tf.shape(word_poly)[0]]
    char_poly, word_poly = tf.split(new_poly, size_splits, axis=0)

    if not disable_rotation_augmentation and tf.random.uniform([]) >= 0.5:
        cx = char_poly[..., 0]
        cy = char_poly[..., 1]
        wx = word_poly[..., 0]
        wy = word_poly[..., 1]

        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
            cx = image_size - cx
            wx = image_size - wx
        elif False and tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_up_down(image)
            diff = tf.stack([0., image_size])
            cy = image_size - cy
            wy = image_size - wy
        elif tf.random.uniform([]) >= 0.5:
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            angle = k * 90

            def rot90(x, y):
                return -y + image_size, x
            def rot180(x, y):
                return -x + image_size, -y + image_size
            def rot270(x, y):
                return y, -x + image_size

            if k == 3:
                cx, cy = rot90(cx, cy)
                wx, wy = rot90(wx, wy)
            if k == 2:
                cx, cy = rot180(cx, cy)
                wx, wy = rot180(wx, wy)
            if k == 1:
                cx, cy = rot270(cx, cy)
                wx, wy = rot270(wx, wy)

            image = tf.image.rot90(image, k)

        char_poly = tf.stack([cx, cy], -1)
        word_poly = tf.stack([wx, wy], -1)

    return image, char_poly, word_poly
