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
    with tf.Graph().as_default() as g:
        if FLAGS.checkpoint:
            checkpoint_prefix = FLAGS.checkpoint
        elif FLAGS.checkpoint_dir:
            checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        #saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            #saver.restore(sess, FLAGS.input)

            tf.keras.backend.set_learning_phase(0)
            tf.keras.backend.set_session(sess)


            model = encoder.create_model(FLAGS.model_name, FLAGS.num_classes)
            image_size = model.image_size
            all_anchors, all_grid_xy, all_ratios = anchors_gen.generate_anchors(image_size, model.output_sizes)

            checkpoint = tf.train.Checkpoint(model=model)

            images = tf.keras.layers.Input(shape=(image_size * image_size * 3), name='input/images_rgb', dtype=tf.uint8)
            images = tf.reshape(images, [-1, image_size, image_size, 3])
            images = normalize_image(images, tf.float32)

            coords_batch, scores_batch, objs_batch, cat_ids_batch = eval_step_logits(model, images, image_size, FLAGS.num_classes, all_anchors, all_grid_xy, all_ratios)

            coords_batch = tf.identity(coords_batch, name='output/coords')
            scores_batch = tf.identity(scores_batch, name='output/scores')
            objs_batch = tf.identity(objs_batch, name='output/objectness')
            cat_ids_batch = tf.identity(cat_ids_batch, name='output/category_ids')

            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched().expect_partial()
            logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['output/coords', 'output/scores', 'output/objectness', 'output/category_ids'])

            output = '{}-{}.frozen.pb'.format(checkpoint_prefix, tf.__version__)
            filename = tf.io.write_graph(output_graph_def, os.path.dirname(output), os.path.basename(output), as_text=False)

            print('Saved graph as {}'.format(os.path.abspath(filename)))

if __name__ == '__main__':
    FLAGS = parser.parse_args()

    tf.app.run()
