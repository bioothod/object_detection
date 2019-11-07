import argparse
import cv2
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

import encoder
import preprocess

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='Number of images to process in a batch')
parser.add_argument('--max_sentence_length', type=int, default=32, help='Maximum sentence lehgth')
parser.add_argument('--num_cpus', type=int, default=6, help='Number of parallel preprocessing jobs')
parser.add_argument('--min_obj_score', type=float, default=0.3, help='Minimal class probability')
parser.add_argument('--output_dir', type=str, required=True, help='Path to directory, where images will be stored')
parser.add_argument('--checkpoint', type=str, help='Load model weights from this file')
parser.add_argument('--checkpoint_dir', type=str, help='Load model weights from the latest checkpoint in this directory')
parser.add_argument('--model_name', type=str, default='efficientnet-b0', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--freeze', action='store_true', help='Save frozen protobuf near checkpoint')
parser.add_argument('--eval_tfrecord_dir', type=str, help='Directory containing evaluation TFRecords')
parser.add_argument('--dataset_type', type=str, choices=['files', 'tfrecords', 'filelist'], default='files', help='Dataset type')
parser.add_argument('filenames', type=str, nargs='*', help='Numeric label : file path')
parser.add_argument('--image_shape', type=str, default='32x128', help='Use this image shape: HxW')
FLAGS = parser.parse_args()

def tf_read_image(filename, image_shape, dtype):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.pad_resize_image(image, image_shape)

    image = tf.cast(image, tf.float32)
    image -= 128.
    image /= 128.


    return filename, image

@tf.function
def eval_step_logits(model, images):
    logits = model(images, training=False)

    # BxTxC -> TxBxC
    logits = tf.transpose(logits, [1, 0, 2])

    #decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=self.max_sentence_len, beam_width=50)

    lengths = tf.ones((tf.shape(images)[0]), dtype=tf.int32) * model.max_sentence_len
    decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=lengths)
    tf.print(decoded)
    decoded = decoded[0]
    dense = tf.sparse.to_dense(decoded, default_value=0)

    return dense

def run_eval(model, dataset, image_shape, num_images):
    num_files = 0
    dump_js = []
    for filenames, images in dataset:
        start_time = time.time()
        sentences_batch = eval_step_logits(model, images)
        logger.info('decoded: {}'.format(sentences_batch.numpy()))
        exit(0)
        num_files += len(filenames)
        time_per_image_ms = (time.time() - start_time) / len(filenames) * 1000

        logger.info('batch images: {}, total_processed: {}, time_per_image: {:.1f} ms'.format(len(filenames), num_files, time_per_image_ms))

        for filename, texts in zip(filenames, sentences_batch):
            filename = str(filename.numpy(), 'utf8')
            texts = texts.numpy()
            #image = cv2.imread(filename)

            base_filename = os.path.basename(filename)

            logger.info('{}: {}'.format(filename, texts))

    return

    json_fn = os.path.join(FLAGS.output_dir, 'results.json')
    logger.info('Saving {} objects into {}'.format(len(dump_js), json_fn))

    with open(json_fn, 'w') as fout:
        json.dump(dump_js, fout)


def run_inference():
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, 'infdet.log'), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    image_shape = [int(d) for d in FLAGS.image_shape.split('x')][:2]

    model = encoder.create_text_recognition_model(FLAGS.model_name, FLAGS.max_sentence_length)

    checkpoint = tf.train.Checkpoint(model=model)

    if FLAGS.checkpoint:
        checkpoint_prefix = FLAGS.checkpoint
    elif FLAGS.checkpoint_dir:
        checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if FLAGS.freeze:
        with tf.compat.v1.Session() as sess:
            images = tf.keras.layers.Input(shape=(image_shape[0] * image_shape[1] * 3), name='input/images_rgb', dtype=tf.uint8)
            images = tf.reshape(images, [-1] + image_shape + [3])
            images -= 128
            images /= 128

            model = encoder.create_model(FLAGS.model_name)

            checkpoint = tf.train.Checkpoint(model=model)

            coords_batch, objs_batch = eval_step_logits(model, images, image_size, all_anchors, all_grid_xy, all_ratios)

            coords_batch = tf.identity(coords_batch, name='output/coords')
            objs_batch = tf.identity(objs_batch, name='output/objectness')

            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched().expect_partial()
            logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['output/coords', 'output/objectness'])

        output = '{}-{}.frozen.pb'.format(checkpoint_prefix, tf.__version__)
        filename = tf.io.write_graph(output_graph_def, os.path.dirname(output), os.path.basename(output), as_text=False)

        print('Saved graph as {}'.format(os.path.abspath(filename)))
        return


    status = checkpoint.restore(checkpoint_prefix)
    status.assert_existing_objects_matched().expect_partial()
    logger.info("Restored from external checkpoint {}".format(checkpoint_prefix))

    num_images = len(FLAGS.filenames)
    dtype = tf.float32

    if FLAGS.dataset_type == 'files':
        ds = tf.data.Dataset.from_tensor_slices((FLAGS.filenames))
        ds = ds.map(lambda fn: tf_read_image(fn, image_shape, dtype), num_parallel_calls=FLAGS.num_cpus)
    elif FLAGS.dataset_type == 'filelist':
        filenames = []
        for fn in FLAGS.filenames:
            with open(fn, 'r') as fin:
                for line in fin:
                    if line[-1] == '\n':
                        line = line[:-1]
                    filenames.append(line)
        ds = tf.data.Dataset.from_tensor_slices((filenames))
        ds = ds.map(lambda fn: tf_read_image(fn, image_shape, dtype), num_parallel_calls=FLAGS.num_cpus)
    elif FLAGS.dataset_type == 'tfrecords':
        logger.fatal('unsupported yet dataset type')

    ds = ds.batch(FLAGS.batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    logger.info('Dataset has been created: num_images: {}, model_name: {}'.format(num_images, FLAGS.model_name))

    run_eval(model, ds, image_shape, FLAGS.output_dir)

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    if not FLAGS.checkpoint and not FLAGS.checkpoint_dir:
        logger.critical('You must provide either checkpoint or checkpoint dir')
        exit(-1)

    try:
        run_inference()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        logger.error("got error: {}".format(e))

        import traceback

        lines = traceback.format_exc().splitlines()
        for l in lines:
            logger.info(l)

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        exit(-1)
