import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

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
parser.add_argument('--image_size', type=int, default=528, help='Input image size')
parser.add_argument('--model_name', type=str, action='store', required=True, help='Model name')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--checkpoint_dir', type=str, help='Load model weights from the latest checkpoint in this directory')
parser.add_argument('--max_ret', type=int, default=100, help='Maximum number of returned boxes')
parser.add_argument('--min_score', type=float, default=0.7, help='Minimal class probability')
parser.add_argument('--min_obj_score', type=float, default=0.3, help='Minimal class probability')
parser.add_argument('--min_size', type=float, default=4, help='Minimal size of the bounding box')
parser.add_argument('--iou_threshold', type=float, default=0.45, help='Minimal IoU threshold for non-maximum suppression')
parser.add_argument('--output_dir', type=str, help='Path to directory, where images will be stored')

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

    class MyModel(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.model = encoder.create_model(FLAGS.model_name, FLAGS.num_classes)

            self.all_anchors, self.all_grid_xy, self.all_ratios = anchors_gen.generate_anchors(FLAGS.image_size, self.model.output_sizes)

            checkpoint = tf.train.Checkpoint(model=self.model)

            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched().expect_partial()
            logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

        @tf.function(input_signature=[tf.TensorSpec([None, FLAGS.image_size * FLAGS.image_size * 3], tf.uint8, name='model_input_images')])
        def __call__(self, inputs):
            images = tf.reshape(inputs, [-1, FLAGS.image_size, FLAGS.image_size, 3])
            images = normalize_image(images, tf.float32)

            best_coords, best_scores, best_objs, best_cat_ids = eval_step_logits(self.model, images, FLAGS.image_size, FLAGS.num_classes,
                    self.all_anchors, self.all_grid_xy, self.all_ratios,
                    FLAGS.min_obj_score, FLAGS.min_score, FLAGS.min_size, FLAGS.max_ret, FLAGS.iou_threshold)

            return best_coords, best_scores, best_objs, best_cat_ids


    model = MyModel()

    logger.info('model with {} backend has been created, image size: {}'.format(FLAGS.model_name, FLAGS.image_size))
    #x = tf.keras.layers.Input(shape=(FLAGS.image_size * FLAGS.image_size * 3), dtype=tf.uint8, name='input/images')
    x = tf.zeros((1, FLAGS.image_size * FLAGS.image_size * 3), dtype=tf.uint8)
    _ = model(x)

    output_dir = '{}_saved_model'.format(checkpoint_prefix)
    tf.saved_model.save(model, output_dir)

    logger.info('Saved model into {}'.format(output_dir))

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(is_dynamic_op=True)

    if False:
        conversion_params = tf.experimental.tensorrt.ConversionParams(
            rewriter_config_template=None,
            max_workspace_size_bytes=tf.experimental.tensorrt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
            precision_mode=tf.experimental.tensorrt.TrtPrecisionMode.FP16,
            minimum_segment_size=3,
            is_dynamic_op=True,
            maximum_cached_engines=1,
            use_calibration=False,
            max_batch_size=100,
            allow_build_at_runtime=True
        )


    converter = trt.TrtGraphConverterV2(input_saved_model_dir=output_dir, conversion_params=conversion_params)
    converter.convert()
    converter.save('{}_trt'.format(output_dir))

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main()
