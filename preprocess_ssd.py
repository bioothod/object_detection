import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


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

def ssd_random_sample_patch(image, labels, polys, bboxes, ratio_list=[0.1, 0.3, 0.5, 0.7, 0.9, 1.], name=None):
    '''ssd_random_sample_patch.
    select one min_iou
    sample _width and _height from [0-width] and [0-height]
    check if the aspect ratio between 0.5-2.
    select left_top point from (width - _width, height - _height)
    check if this bbox has a min_iou with all ground_truth bboxes
    keep ground_truth those center is in this sampled patch, if none then try again
    '''
    def sample_width_height(width, height):
        with tf.name_scope('sample_width_height'):
            index = 0
            max_attempt = 10
            sampled_width, sampled_height = width, height

            def condition(index, sampled_width, sampled_height, width, height):
                return tf.logical_or(tf.logical_and(tf.logical_or(tf.greater(sampled_width, sampled_height * 2),
                                                                tf.greater(sampled_height, sampled_width * 2)),
                                                    tf.less(index, max_attempt)),
                                    tf.less(index, 1))

            def body(index, sampled_width, sampled_height, width, height):
                sampled_width = tf.random.uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] * width
                sampled_height = tf.random.uniform([1], minval=0.3, maxval=0.999, dtype=tf.float32)[0] *height

                return index+1, sampled_width, sampled_height, width, height

            [index, sampled_width, sampled_height, _, _] = tf.while_loop(condition, body,
                                               [index, sampled_width, sampled_height, width, height], parallel_iterations=4, back_prop=False, swap_memory=True)

            return tf.cast(sampled_width, tf.int32), tf.cast(sampled_height, tf.int32)

    def jaccard_with_anchors(roi, bboxes):
        with tf.name_scope('jaccard_with_anchors'):
            int_ymin = tf.maximum(roi[0], bboxes[:, 0])
            int_xmin = tf.maximum(roi[1], bboxes[:, 1])
            int_ymax = tf.minimum(roi[2], bboxes[:, 2])
            int_xmax = tf.minimum(roi[3], bboxes[:, 3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            union_vol = (roi[3] - roi[1]) * (roi[2] - roi[0]) + ((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) - inter_vol)
            jaccard = tf.math.divide(inter_vol, union_vol)
            return jaccard

    def areas(bboxes):
        with tf.name_scope('bboxes_areas'):
            vol = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            return vol

    @tf.function
    def check_roi_center(width, height, bboxes):
        with tf.name_scope('check_roi_center'):
            max_attempts = 20
            roi = [0., 0., 0., 0.]
            float_width = tf.cast(width, tf.float32)
            float_height = tf.cast(height, tf.float32)
            mask = tf.cast(tf.zeros(tf.shape(bboxes)[0], dtype=tf.uint8), tf.bool)
            center_x, center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2

            for index in tf.range(max_attempts):
                sampled_width, sampled_height = sample_width_height(float_width, float_height)

                x = tf.random.uniform([], minval=0, maxval=width - sampled_width, dtype=tf.int32)
                y = tf.random.uniform([], minval=0, maxval=height - sampled_height, dtype=tf.int32)

                roi = [tf.cast(y, tf.float32) / float_height,
                      tf.cast(x, tf.float32) / float_width,
                      tf.cast(y + sampled_height, tf.float32) / float_height,
                      tf.cast(x + sampled_width, tf.float32) / float_width]

                mask_min = tf.logical_and(tf.greater(center_y, roi[0]), tf.greater(center_x, roi[1]))
                mask_max = tf.logical_and(tf.less(center_y, roi[2]), tf.less(center_x, roi[3]))
                mask = tf.logical_and(mask_min, mask_max)

                if tf.reduce_sum(tf.cast(mask, tf.int32)) >= 1:
                    break

            return roi, mask

    @tf.function
    def check_roi_overlap(width, height, labels, polys, bboxes, min_iou):
        with tf.name_scope('check_roi_overlap'):
            max_attempt = 50
            roi = [0., 0., 1., 1.]
            mask = tf.cast(tf.zeros(tf.shape(bboxes)[0], dtype=tf.uint8), tf.bool)

            for index in tf.range(max_attempt):
                roi, mask = check_roi_center(width, height, bboxes)
                mask_labels = tf.boolean_mask(labels, mask)
                if tf.shape(mask_labels)[0] == 0:
                    continue

                mask_bboxes = tf.boolean_mask(bboxes, mask)

                jac = jaccard_with_anchors(roi, mask_bboxes)
                s = tf.reduce_sum(tf.cast(jac < min_iou, tf.int32))
                if s == 0:
                    break

            mask_labels = tf.boolean_mask(labels, mask)
            mask_polys = tf.boolean_mask(polys, mask)
            mask_bboxes = tf.boolean_mask(bboxes, mask)

            if tf.shape(mask_labels)[0] > 0:
                height = tf.cast(height, tf.float32)
                width = tf.cast(width, tf.float32)

                return (tf.cast([roi[0] * height, roi[1] * width, (roi[2] - roi[0]) * height, (roi[3] - roi[1]) * width], tf.int32), mask_labels, mask_polys, mask_bboxes)

            return (tf.cast([0, 0, height, width], tf.int32), labels, polys, bboxes)


    def sample_patch(image, labels, polys, bboxes, min_iou):
        with tf.name_scope('sample_patch'):
            height, width, depth = _ImageDimensions(image, rank=3)

            roi_slice_range, mask_labels, mask_polys, mask_bboxes = check_roi_overlap(width, height, labels, polys, bboxes, min_iou)

            scale = tf.cast(tf.stack([height, width, height, width]), mask_bboxes.dtype)
            mask_bboxes = mask_bboxes * scale

            # Add offset.
            offset = tf.cast(tf.stack([roi_slice_range[0], roi_slice_range[1], roi_slice_range[0], roi_slice_range[1]]), mask_bboxes.dtype)
            mask_bboxes = mask_bboxes - offset
            mask_polys = mask_polys - offset[..., :2]

            cliped_ymin = tf.maximum(0., mask_bboxes[:, 0])
            cliped_xmin = tf.maximum(0., mask_bboxes[:, 1])
            cliped_ymax = tf.minimum(tf.cast(roi_slice_range[2], tf.float32), mask_bboxes[:, 2])
            cliped_xmax = tf.minimum(tf.cast(roi_slice_range[3], tf.float32), mask_bboxes[:, 3])

            mask_bboxes = tf.stack([cliped_ymin, cliped_xmin, cliped_ymax, cliped_xmax], axis=-1)
            # Rescale to target dimension.
            scale_bboxes = tf.cast(tf.stack([roi_slice_range[2], roi_slice_range[3],
                                      roi_slice_range[2], roi_slice_range[3]]), mask_bboxes.dtype)
            scale_polys = tf.cast(tf.stack([roi_slice_range[3], roi_slice_range[2]]), mask_bboxes.dtype)

            return tf.cond(tf.logical_or(tf.less(roi_slice_range[2], 1), tf.less(roi_slice_range[3], 1)),
                        lambda: (image, labels, polys, bboxes),
                        lambda: (tf.slice(image, [roi_slice_range[0], roi_slice_range[1], 0], [roi_slice_range[2], roi_slice_range[3], -1]),
                                    mask_labels, mask_polys / scale_polys, mask_bboxes / scale_bboxes))

    with tf.name_scope('ssd_random_sample_patch'):
        image = tf.convert_to_tensor(image, name='image')

        min_iou_list = tf.convert_to_tensor(ratio_list)
        samples_min_iou = tf.random.categorical(tf.math.log([[1. / len(ratio_list)] * len(ratio_list)]), 1)

        sampled_min_iou = min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]

        return tf.cond(tf.less(sampled_min_iou, 1.), lambda: sample_patch(image, labels, polys, bboxes, sampled_min_iou), lambda: (image, labels, polys, bboxes))

def ssd_random_expand(image, polys, bboxes, ratio=2., name=None):
    with tf.name_scope('ssd_random_expand'):
        image = tf.convert_to_tensor(image, name='image')
        if image.get_shape().ndims != 3:
            raise ValueError('\'image\' must have 3 dimensions.')

        height, width, depth = _ImageDimensions(image, rank=3)

        float_height, float_width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

        canvas_width, canvas_height = tf.cast(float_width * ratio, tf.int32), tf.cast(float_height * ratio, tf.int32)

        mean_color_of_image = [_R_MEAN/255., _G_MEAN/255., _B_MEAN/255.]#tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

        x = tf.random.uniform([], minval=0, maxval=canvas_width - width, dtype=tf.int32)
        y = tf.random.uniform([], minval=0, maxval=canvas_height - height, dtype=tf.int32)

        paddings = tf.convert_to_tensor([[y, canvas_height - height - y], [x, canvas_width - width - x]])

        big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values = mean_color_of_image[0]),
                              tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values = mean_color_of_image[1]),
                              tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values = mean_color_of_image[2])], axis=-1)

        scale_bboxes = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
        scale_polys = tf.cast(tf.stack([width, height]), bboxes.dtype)
        absolute_bboxes = bboxes * scale_bboxes + tf.cast(tf.stack([y, x, y, x]), bboxes.dtype)
        absolute_polys = polys * scale_polys + tf.cast(tf.stack([x, y]), bboxes.dtype)

        bboxes = absolute_bboxes / tf.cast(tf.stack([canvas_height, canvas_width, canvas_height, canvas_width]), bboxes.dtype)
        polys = absolute_polys / tf.cast(tf.stack([canvas_width, canvas_height]), bboxes.dtype)

        return big_canvas, polys, bboxes

def ssd_random_sample_patch_wrapper(image, labels, polys, bboxes):
    with tf.name_scope('ssd_random_sample_patch_wrapper'):
        orig_image, orig_labels, orig_polys, orig_bboxes = image, labels, polys, bboxes
        def check_bboxes(bboxes):
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            return tf.logical_and(tf.logical_and(areas < 0.9, areas > 0.001),
                                  tf.logical_and((bboxes[:, 3] - bboxes[:, 1]) > 0.025, (bboxes[:, 2] - bboxes[:, 0]) > 0.025))

        index = 0
        max_attempt = 3
        def condition(index, image, labels, polys, bboxes, orig_image, orig_labels, orig_polys, orig_bboxes):
            return tf.logical_or(tf.logical_and(tf.reduce_sum(tf.cast(check_bboxes(bboxes), tf.int64)) < 1, tf.less(index, max_attempt)), tf.less(index, 1))

        def body(index, image, labels, polys, bboxes, orig_image, orig_labels, orig_polys, orig_bboxes):
            image, polys, bboxes = tf.cond(tf.random.uniform([], minval=0., maxval=1., dtype=tf.float32) < 0.5,
                            lambda: (orig_image, orig_polys, orig_bboxes),
                            lambda: ssd_random_expand(orig_image, orig_polys, orig_bboxes, tf.random.uniform([1], minval=1.1, maxval=4., dtype=tf.float32)[0]))
            # Distort image and bounding boxes.
            random_sample_image, labels, polys, bboxes = ssd_random_sample_patch(image, orig_labels, polys, bboxes, ratio_list=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.])
            random_sample_image.set_shape([None, None, 3])
            return index+1, random_sample_image, labels, polys, bboxes, orig_image, orig_labels, orig_polys, orig_bboxes

        [index, image, labels, polys, bboxes, orig_image, orig_labels, orig_polys, orig_bboxes] = tf.while_loop(condition, body,
                                                                                                    [index,  image, labels, polys, bboxes, orig_image, orig_labels, orig_polys, orig_bboxes],
                                                                                                    parallel_iterations=8,
                                                                                                    shape_invariants=[tf.TensorShape([]),
                                                                                                        tf.TensorShape([None, None, 3]),
                                                                                                        labels.get_shape(), polys.get_shape(), bboxes.get_shape(),
                                                                                                        tf.TensorShape([None, None, 3]),
                                                                                                        orig_labels.get_shape(), orig_polys.get_shape(), orig_bboxes.get_shape()],
                                                                                                    back_prop=False,
                                                                                                    swap_memory=True)

        valid_mask = check_bboxes(bboxes)
        labels, polys, bboxes = tf.boolean_mask(labels, valid_mask), tf.boolean_mask(polys, valid_mask), tf.boolean_mask(bboxes, valid_mask)
        return tf.cond(tf.less(index, max_attempt),
                    lambda : (image, labels, polys, bboxes),
                    lambda : (orig_image, orig_labels, orig_polys, orig_bboxes))

def _mean_image_subtraction(image, means):
    ndims = image.get_shape().ndims
    if ndims != 3 and ndims != 4:
        raise ValueError('Input must be of size [?, height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means)={} must match the number of channels={}'.format(len(means), num_channels))

    channels = tf.split(axis=-1, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=-1, values=channels)

def denormalize_image(image):
    means=[_R_MEAN, _G_MEAN, _B_MEAN]
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    image = tf.concat(axis=2, values=channels)
    return image

def normalize_image(image):
    #image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    image -= [_R_MEAN, _G_MEAN, _B_MEAN]
    return image

def random_flip_left_right(image, bboxes):
    with tf.name_scope('random_flip_left_right'):
        uniform_random = tf.random.uniform([], 0, 1.0)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        result = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
        # Flip bboxes.
        mirror_bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                                  bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        bboxes = tf.cond(mirror_cond, lambda: mirror_bboxes, lambda: bboxes)
        return result, bboxes

def preprocess_for_train(image, labels, polys, bboxes, out_shape, data_format='channels_last', scope='ssd_preprocessing_train', output_rgb=True):
    """Preprocesses the given image for training.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels: A `Tensor` containing all labels for all bboxes of this image.
      bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
      out_shape: The height and width of the image after preprocessing.
      data_format: The data_format of the desired output image.
    Returns:
      A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        orig_dtype = image.dtype

        if False:
            # Randomly distort the colors. There are 4 ways to do it.
            image = apply_with_random_selector(image,
                                                  lambda x, ordering: distort_color(x, ordering, True),
                                                  num_cases=4)

        image, labels, polys, bboxes = ssd_random_sample_patch_wrapper(image, labels, polys, bboxes)
        final_image = tf.image.resize(image, out_shape, method=tf.image.ResizeMethod.BILINEAR)

        final_image.set_shape(out_shape + [3])
        if not output_rgb:
            image_channels = tf.unstack(final_image, axis=-1, name='split_rgb')
            final_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
        if data_format == 'channels_first':
            final_image = tf.transpose(final_image, perm=(2, 0, 1))
        return final_image, labels, polys, bboxes

def preprocess_for_eval(image, out_shape, data_format='channels_last', scope='ssd_preprocessing_eval', output_rgb=True):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      out_shape: The height and width of the image after preprocessing.
      data_format: The data_format of the desired output image.
    Returns:
      A preprocessed image.
    """
    with tf.name_scope(scope):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, out_shape, method=tf.image.ResizeMethod.BILINEAR)
        image.set_shape(out_shape + [3])

        if not output_rgb:
            image_channels = tf.unstack(image, axis=-1, name='split_rgb')
            image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
        # Image data format.
        if data_format == 'channels_first':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image

def preprocess_image(image, labels, bboxes, out_shape, is_training=False, data_format='channels_last', output_rgb=True):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels: A `Tensor` containing all labels for all bboxes of this image.
      bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
      out_shape: The height and width of the image after preprocessing.
      is_training: Wether we are in training phase.
      data_format: The data_format of the desired output image.

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, out_shape, data_format=data_format, output_rgb=output_rgb)
    else:
        return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)

