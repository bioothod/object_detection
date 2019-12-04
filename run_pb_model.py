import argparse

import tensorflow as tf

from tensorflow.core.framework import graph_pb2

import efficientnet as efn
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_pb', required=True, type=str, help='Input protobuf file')
parser.add_argument('--model_name', required=True, type=str, help='Model name to infer image size from')
parser.add_argument('filenames', nargs='+', type=str, help='Input files')
FLAGS = parser.parse_args()

def tf_read_image(filename, image_size):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = preprocess.pad_resize_image(image, [image_size, image_size])
    image = tf.reshape(image, [-1])
    image = tf.cast(image, tf.uint8)

    return filename, image

def main():
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        input_graph_def = graph_pb2.GraphDef()
        with open(FLAGS.input_pb, 'rb') as fin:
            input_graph_def.ParseFromString(fin.read())

        image_size = efn.efficientnet_params(FLAGS.model_name)[2]

        ds = tf.data.Dataset.from_tensor_slices((FLAGS.filenames))
        ds = ds.map(lambda fn: tf_read_image(fn, image_size))
        ds = ds.batch(3)

        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        filenames_op, images_op = iterator.get_next()

        output_names = ['output/coords:0', 'output/objectness:0', 'output/scores:0', 'output/category_ids:0']

        input_map = {
            'input/images_rgb:0': images_op,
        }

        output_nodes = tf.import_graph_def(input_graph_def, return_elements=output_names, input_map=input_map)
        #coords_op, objs_op, scores_op, cat_ids_op = output_nodes

        for res in sess.run([filenames_op, images_op] + output_nodes):
            print(res)
            #filenames, coords, objs, scores, cat_ids = res
            #print('{}: {}'.format(filenames, objs))

if __name__ == '__main__':
    main()
