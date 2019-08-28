import os
import time

import numpy as np
import tensorflow as tf

import preprocess
import unet

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
def find_centroids(corner_mask):
    max_value = tf.reduce_max(corner_mask)
    points = tf.where(corner_mask > max_value * 0.7)

    num_clusters = 15
    centroids = choose_random_centroids(points, num_clusters)

    for i in range(10):
        nearest_indices = assign_to_nearest(points, centroids)
        centroids = update_centroids(points, nearest_indices, num_clusters)

    return centroids

class Inference:
    def __init__(self, model_name, checkpoint, num_classes=5, centrod_max_distance_to_sqeeze=15, centroid_max_value_threshold=0.7, centroid_num_clisters=15):
        self.centroid_max_value_threshold = centroid_max_value_threshold
        self.centroid_num_clisters = centroid_num_clisters
        self.centrod_max_distance_to_sqeeze = centrod_max_distance_to_sqeeze * centrod_max_distance_to_sqeeze

        self.create_model(model_name, num_classes, checkpoint)

    def create_model(self, model_name, num_classes, checkpoint):
        params = {
            'num_classes': None, # we are creaging a base model which only extract features and does not perform classification
            'data_format': 'channels_last',
            'relu_fn': tf.nn.swish
        }
        base_model, model, image_size = unet.create_model(params, tf.float32, model_name, num_classes)
        self.model = model
        self.image_size = image_size

        print('image size: {}'.format(self.image_size))

        self.checkpoint = tf.train.Checkpoint(model=model)
        self.checkpoint.restore(checkpoint)
        #status.expect_partial()

    @tf.function
    def eval_step(self, image):
        image = preprocess.processing_function(image, self.image_size, self.image_size, False)

        image = tf.expand_dims(image, 0)
        logits = self.model(image, training=False)
        masks = tf.nn.softmax(logits, axis=-1)
        masks = tf.squeeze(masks, 0)
        return masks

    def corners_for_image(self, image):
        mask = self.eval_step(image)

        corner_layer = 4
        corner_mask_tensor = mask[:, :, corner_layer]

        centers = find_centroids(corner_mask_tensor).numpy()

        new_centroids = []
        new_converted_centroids = []
        for c in centers:
            skip = False
            for o in new_centroids:
                dist = np.sum(np.square(c - o))
                if dist < self.centrod_max_distance_to_sqeeze:
                    skip = True

            if not skip:
                # convert to [x, y] from [y, x]
                new_centroids.append(c)
                new_converted_centroids.append([c[1], c[0]])

        return mask, new_converted_centroids

    def generate(self, image_source):
        for image_id, image in image_source:
            yield image_id, self.corners_for_image(image)

def generate_image(filename, image, mask, centers, data_dir):
    filename = os.path.basename(filename)
    image_id = os.path.splitext(filename)[0]

    dst = '{}/{}.png'.format(data_dir, image_id)
    image_draw.draw_im_segm(image, [mask.numpy()], centers, dst)

if __name__ == '__main__':
    import argparse
    import cv2
    import logging

    import image as image_draw

    logger = logging.getLogger('segmentation')
    logger.propagate = False

    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    __handler = logging.StreamHandler()
    __handler.setFormatter(__fmt)
    logger.addHandler(__handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory where to store masked images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Load model weights from this file')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
    FLAGS = parser.parse_args()

    inf = Inference(FLAGS.model_name, FLAGS.checkpoint)

    def get_image():
        filenames = [os.path.join(FLAGS.data_dir, fn) for fn in os.listdir(FLAGS.data_dir) if os.path.splitext(fn.lower())[-1] in ['.png', '.jpg', '.jpeg']]
        for fn in filenames:
            image = cv2.imread(fn)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.uint8)

            yield fn, image

    os.makedirs(FLAGS.dst_dir, exist_ok=True)
    for image_id, image in inf.generate(get_image():
        mask, centers = inf.corners_for_image(image)
        image = cv2.resize(image, (inf.image_size, inf.image_size))
        generate_image(filename, image, mask, centers, FLAGS.dst_dir)
