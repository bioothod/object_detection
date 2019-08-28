import argparse
import logging
import os
import time

import numpy as np
import tensorflow as tf

import efficientnet
import image as image_draw
import preprocess
import unet

logger = logging.getLogger('segmentation')
logger.propagate = False

def generate_images(filenames, images, masks, centroids, data_dir):
    vgg_means = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)

    for filename, image, mask, centers in zip(filenames, images, masks, centroids):
        filename = str(filename)
        filename = os.path.basename(filename)
        image_id = os.path.splitext(filename)[0]

        image = image.numpy() * 128. + 128
        image = image.astype(np.uint8)

        dst = '{}/{}.png'.format(data_dir, image_id)
        image_draw.draw_im_segm(image, [mask.numpy()], centers, dst)

@tf.function
def basic_preprocess(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.processing_function(image, image_size, image_size, False, dtype)

    return filename, image

@tf.function
def load_image(datapoint, image_size, is_training, dtype):
    image = datapoint['image']
    mask = datapoint['segmentation_mask']
    filename = datapoint['file_name']

    image = preprocess.processing_function(image, image_size, image_size, is_training, dtype)
    mask = preprocess.try_resize(mask, image_size, image_size) - 1

    if tf.random.uniform(()) > 0.5 and is_training:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    return filename, image, mask

def choose_random_centroids(samples, n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random.shuffle(tf.range(0, n_samples))
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

def assign_to_nearest(samples, centroids):
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    nearest_indices = mins
    return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.cast(nearest_indices, tf.int32)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

@tf.function
def find_centroids(filename, corner_mask):
    filename = str(filename)

    max_value = tf.reduce_max(corner_mask)
    points = tf.where(corner_mask > max_value * 0.7)

    num_clusters = 15
    centroids = choose_random_centroids(points, num_clusters)

    for i in range(10):
        nearest_indices = assign_to_nearest(points, centroids)
        centroids = update_centroids(points, nearest_indices, num_clusters)

    return centroids

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory where to store masked images')
    parser.add_argument('--dataset', type=str, choices=['card_images', 'oxford_pets'], default='images', help='Dataset type')
    parser.add_argument('--num_cpus', type=int, default=32, help='Number of preprocessing processes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Load model weights from this file')
    parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch.')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
    parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
    FLAGS = parser.parse_args()

    dtype = tf.float32

    params = {
        'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
        'data_format': FLAGS.data_format,
        'relu_fn': tf.nn.swish
    }

    if FLAGS.dataset == 'card_images':
        num_classes = 5
    elif FLAGS.dataset == 'oxford_pets':
        num_classes = 3

    base_model, model, image_size = unet.create_model(params, dtype, FLAGS.model_name, num_classes)

    if FLAGS.dataset == 'card_images':
        filenames = [os.path.join(FLAGS.data_dir, fn) for fn in os.listdir(FLAGS.data_dir) if os.path.splitext(fn.lower())[-1] in ['.png', '.jpg', '.jpeg']]

        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(lambda filename: basic_preprocess(filename, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)
    elif FLAGS.dataset == 'oxford_pets':
        import tensorflow_datasets as tfds

        dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True, data_dir=FLAGS.data_dir)

        dataset = dataset['test'].map(lambda datapoint: load_image(datapoint, image_size, False, dtype), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)


    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint)
    status.expect_partial()
    logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))

    @tf.function
    def eval_step(images):
        logits = model(images, training=False)
        masks = tf.nn.softmax(logits, axis=-1)
        return masks

    os.makedirs(FLAGS.dst_dir, exist_ok=True)

    corner_layer = 4

    start_time = time.time()
    total_images = 0
    for t in dataset:
        filenames = t[0]
        images = t[1]

        masks = eval_step(images)
        total_images += len(filenames)


        centroids = []
        for filename, mask in zip(filenames, masks):
            corner_mask_tensor = mask[:, :, corner_layer]
            
            corner_mask = mask[:, :, corner_layer].numpy()

            #max_value = corner_mask.max()
            #idx = corner_mask > max_value * 0.8
            #wh = np.where(idx)
            #points = np.stack(wh, axis=1)
            #logger.info('{}: max: {}, selected shape: {}, points shape: {}'.format(filename, max_value, corner_mask.shape, points.shape))

            centers = find_centroids(filename, corner_mask_tensor).numpy()

            new_centroids = []
            for c in centers:
                skip = False
                for o in new_centroids:
                    dist = np.abs(c - o)
                    if np.any(dist < 10):
                        skip = True

                if not skip:
                    new_centroids.append(c)

            centroids.append(new_centroids)

            #logger.info('{}: max: {}, selected shape: {}, points shape: {}, centroids: {}'.format(filename, max_value, corner_mask.shape, points.shape, centers))

        generate_images(filenames, images, masks, centroids, FLAGS.dst_dir)
        logger.info('saved {}/{} images'.format(len(filenames), total_images))

    dur = time.time() - start_time
    logger.info('processed {} images, time: {:.2f} seconds, {:.1f} ms per image'.format(total_images, dur, dur / total_images * 1000.))
