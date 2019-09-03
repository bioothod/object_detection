import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

import anchor
import image as image_draw
import preprocess
import ssd

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--num_classes', type=int, required=True, help='Number of the output classes in the model')
parser.add_argument('--max_ret', type=int, default=100, help='Maximum number of returned boxes')
parser.add_argument('--min_score', type=float, default=0.7, help='Minimal class probability')
parser.add_argument('--output_dir', type=str, required=True, help='Path to directory, where images will be stored')
parser.add_argument('--checkpoint', type=str, required=True, help='Load model weights from this file')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')
FLAGS = parser.parse_args()

min_size = 10

def tf_read_image(filename, image_size, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.prepare_image_for_evaluation(image, image_size, image_size, dtype)
    return filename, image

def per_image_supression(anchors, num_classes):
    coords, classes = anchors
    classes = tf.nn.softmax(classes, axis=-1)

    logger.info('supression: coords: {}, classes: {}'.format(coords, classes))

    max_probs = tf.reduce_max(classes, axis=1)
    index = tf.where(max_probs > FLAGS.min_score)
    classes = tf.gather(classes, index)
    classes = tf.squeeze(classes, 1)
    coords = tf.gather(coords, index)
    coords = tf.squeeze(coords, 1)
    logger.info('supression: max_probs reduction: coords: {}, classes: {}'.format(coords, classes))

    ret_coords, ret_scores, ret_cat_ids = [], [], []
    for cat_id in range(1, num_classes):
        class_scores = classes[:, cat_id]

        index = tf.where(class_scores > FLAGS.min_score)
        selected_scores = tf.gather(class_scores, index)
        selected_scores = tf.squeeze(selected_scores, 1)

        selected_coords = tf.gather(coords, index)
        selected_coords = tf.squeeze(selected_coords, 1)

        selected_indexes = tf.image.non_max_suppression(selected_coords, selected_scores, FLAGS.max_ret, score_threshold=FLAGS.min_score)
        selected_coords = tf.gather(selected_coords, selected_indexes)
        selected_scores = tf.gather(selected_scores, selected_indexes)

        ret_coords.append(selected_coords)
        ret_scores.append(selected_scores)

        num = tf.shape(selected_scores)[0]
        tile = tf.tile([cat_id], [num])
        ret_cat_ids.append(tile)

    ret_coords = tf.concat(ret_coords, 0)
    ret_scores = tf.concat(ret_scores, 0)
    ret_cat_ids = tf.concat(ret_cat_ids, 0)

    best_scores, best_index = tf.nn.top_k(ret_scores, tf.minimum(FLAGS.max_ret, tf.shape(ret_scores)[0]), sorted=True)

    best_coords = tf.gather(ret_coords, best_index)
    best_cat_ids = tf.gather(ret_cat_ids, best_index)

    to_add = (FLAGS.max_ret - tf.shape(best_scores)[0])

    best_coords = tf.pad(best_coords, [[0, to_add], [0, 0]] , 'CONSTANT')
    best_scores = tf.pad(best_scores, [[0, to_add]], 'CONSTANT')
    best_cat_ids = tf.pad(best_cat_ids, [[0, to_add]], 'CONSTANT')


    logger.info('ret_coords: {}, ret_scores: {}, ret_cat_ids: {}, best_index: {}, best_scores: {}, best_coords: {}, best_cat_ids: {}'.format(
        ret_coords, ret_scores, ret_cat_ids, best_index, best_scores, best_coords, best_cat_ids))

    return best_coords, best_scores, best_cat_ids

@tf.function
def eval_step_logits(model, images, num_classes):
    logits = model(images, training=False)

    return tf.map_fn(lambda anchor: per_image_supression(anchor, num_classes),
                     logits,
                     parallel_iterations=FLAGS.num_cpus,
                     back_prop=False,
                     infer_shape=False,
                     dtype=(tf.float32, tf.float32, tf.int32))

def run_eval(model, dataset, num_images, num_classes, dst_dir):
    num_files = 0
    for filenames, images in dataset:
        coords_batch, scores_batch, cat_ids_batch = eval_step_logits(model, images, num_classes)
        logger.info('batch: shapes: coords: {}, scores: {}, cat_ids: {}'.format(coords_batch.shape, scores_batch.shape, cat_ids_batch.shape))

        num_files += len(filenames)

        for filename, coords, scores, cat_ids in zip(filenames, coords_batch, scores_batch, cat_ids_batch):
            filename = str(filename)

            logger.info('{}: coords: {}, scores: {}, cat_ids: {}'.format(filename, coords.shape, scores.shape, cat_ids.shape))

            anns = []

        return
            #for cat_id in range(num_classes)

def train():
    dtype = tf.float32
    base_model, image_size, feature_shapes = ssd.create_base_model(dtype, FLAGS.model_name)
    np_anchor_boxes, np_anchor_areas, anchor_layers = anchor.create_anchors(image_size, feature_shapes)
    model = ssd.create_model(base_model, image_size, FLAGS.num_classes, anchor_layers, feature_shapes)

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint)
    status.assert_existing_objects_matched()
    logger.info("Restored from external checkpoint {}".format(FLAGS.checkpoint))


    ds = tf.data.Dataset.from_tensor_slices((FLAGS.filenames))
    ds = ds.map(lambda fn: tf_read_image(fn, image_size, dtype), num_parallel_calls=FLAGS.num_cpus)
    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()

    num_images = len(FLAGS.filenames)
    logger.info('Dataset has been created: num_images: {}, num_classes: {}, model_name: {}'.format(num_images, FLAGS.num_classes, FLAGS.model_name))

    logger.info('anchor_layers: {}, feature_shapes: {}'.format(anchor_layers, feature_shapes))
    run_eval(model, ds, num_images, FLAGS.num_classes, FLAGS.output_dir)

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    try:
        train()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
