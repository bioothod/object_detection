import autoaugment

import tensorflow as tf

def pad_resize_image(image, dims):
    image = tf.image.resize(image, [32, 128], preserve_aspect_ratio=True)

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[sy, syd - sy], [sx, sxd - sx], [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=128)
    image = tf.cast(image, dtype=tf.uint8)
    return image

def preprocess_image(image, image_shape, is_training):
    imgh = tf.cast(tf.shape(image)[0], tf.float32)
    imgw = tf.cast(tf.shape(image)[1], tf.float32)

    scale = tf.random.uniform([], minval=1, maxval=3)

    image = tf.image.resize(image, [int(imgh*scale), int(imgw*scale)], preserve_aspect_ratio=False)
    imgh = tf.shape(image)[0]
    imgw = tf.shape(image)[1]

    target_height, target_width = image_shape

    hpad = target_height - imgh
    hpad = tf.maximum(hpad, 0)
    offset_height = tf.cast(hpad / 2, tf.int32)
    target_height = tf.maximum(target_height, imgh)

    wpad = target_width - imgw
    wpad = tf.maximum(wpad, 0)
    offset_width = tf.cast(wpad / 2, tf.int32)
    target_width = tf.maximum(target_width, imgw)

    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    image = tf.image.resize(image, image_shape, preserve_aspect_ratio=False)

    if is_training:
        image = tf.cast(image, tf.uint8)
        image = autoaugment.distort_image_with_autoaugment(image, 'v0')

    image = tf.cast(image, tf.float32)
    image -= 128.
    image /= 128.
