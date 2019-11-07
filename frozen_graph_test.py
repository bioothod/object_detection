import tensorflow as tf
import numpy as np

import argparse
import cv2
import math
import time

from google.protobuf import text_format


parser = argparse.ArgumentParser()
parser.add_argument('--frozen_pb', type=str, required=True, help='Frozen graph filename')
parser.add_argument('filenames', type=str, nargs='*')

FLAGS = parser.parse_args()

def main():
    with tf.Graph().as_default() as graph:
        graph_def = tf.compat.v1.GraphDef()

        with open(FLAGS.frozen_pb, "rb") as f:
            graph_def.ParseFromString(f.read())
            #text_format.Merge(f.read(), graph_def)


        tf.import_graph_def(graph_def, name='')

        coords_op = graph.get_tensor_by_name('output/coords:0')
        objectness_op = graph.get_tensor_by_name('output/objectness:0')
        scores_op = graph.get_tensor_by_name('output/scores:0')
        cat_ids_op = graph.get_tensor_by_name('output/category_ids:0')

        input_tensor = graph.get_tensor_by_name('input/images_rgb:0')
        dim = int(input_tensor.shape[1])
        image_size = int(math.sqrt(dim / 3))

        print('resizing images to {}x{}x3'.format(image_size, image_size))

        with tf.Session(graph=graph) as sess:
            for filename in FLAGS.filenames:
                start_time = time.time()

                input_images = []

                image = cv2.imread(filename)
                image = cv2.resize(image, (image_size, image_size))

                image = np.expand_dims(image, 0)
                image = np.reshape(image, [-1, image_size*image_size*3])

                tf_start_time = time.time()
                results = sess.run([coords_op, objectness_op, scores_op, cat_ids_op], feed_dict={input_tensor: image})

                coords_batch, objectness_batch, scores_batch, cat_ids_batch = results
                tf_end_time = time.time()

                decode_time = (tf_start_time - start_time) * 1000
                tf_process_time = (tf_end_time - tf_start_time) * 1000

                print('{}: decode time: {:.1f} ms, tf processing time: {:.1f} ms'.format(
                            filename,
                            decode_time, tf_process_time))

                for bbs, objs, scores, cat_ids in zip(coords_batch, objectness_batch, scores_batch, cat_ids_batch):
                    for bb, obj, score, cat_id in zip(bbs, objs, scores, cat_ids):
                        if score == 0 or obj < 0.3:
                            break

                        print('{}: {}, obj: {:.3f}, score: {:.3f}, cat_id: {}, decode time: {:.1f} ms, tf processing time: {:.1f} ms'.format(
                            filename,
                            bb, obj, score, cat_id,
                            decode_time, tf_process_time))


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{:0.3f}'.format, 'int': '{:3d}'.format}, linewidth=250, suppress=True)

    main()
