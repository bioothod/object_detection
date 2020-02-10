import argparse
import logging
import os

import numpy as np
import tensorflow as tf

import anchors_gen
import encoder
import preprocess_ssd

from infdet import eval_step_logits

logger = logging.getLogger('freeze')

logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, action='store', required=True, help='Number of classes')
parser.add_argument('--model_name', type=str, action='store', required=True, help='Model name')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--checkpoint_dir', type=str, help='Load model weights from the latest checkpoint in this directory')
parser.add_argument('--output_dir', type=str, required=True, help='Path to directory, where images will be stored')

def normalize_image(image, dtype):
    image = tf.cast(image, dtype)

    image = preprocess_ssd.normalize_image(image)
    return image

def main(argv=None):
    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    else:
        logger.error('You have to specify either checkpoint or checkpoint directory')
        exit(-1)

    model = encoder.create_model(FLAGS.model_name, FLAGS.num_classes)
    image_size = model.image_size
    all_anchors, all_grid_xy, all_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

    checkpoint = tf.train.Checkpoint(model=model)

    status = checkpoint.restore(checkpoint_prefix)
    status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    @tf.function(input_signature=[tf.TensorSpec([image_size * image_size * 3], tf.uint8)])
    def control_flow(images):
        images = tf.reshape(images, [-1, image_size, image_size, 3])
        images = normalize_image(images, tf.float32)

        return eval_step_logits(model, images, image_size, FLAGS.num_classes, all_anchors, all_grid_xy, all_ratios)

    dummy_image = tf.zeros((image_size * image_size * 3), dtype=tf.uint8)
    control_flow(dummy_image)

    to_export = tf.Module()
    to_export.model = model
    to_export.control_flow = control_flow
    tf.saved_model.save(to_export, FLAGS.output_dir)

    logger.info('Saved model into {}'.format(FLAGS.output_dir))

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main()
