import argparse

import tensorflow as tf

from tensorflow.core.framework import graph_pb2

import efficientnet as efn

parser = argparse.ArgumentParser()
parser.add_argument('--input_pb', required=True, type=str, help='Input protobuf file')
parser.add_argument('--model_name', required=True, type=str, help='Model name to infer image size from')
parser.add_argument('filenames', nargs='+', type=str, help='Input files')
FLAGS = parser.parse_args()

def tf_read_image(filename, image_size):
    dtype = tf.float32

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    orig_image_height = tf.cast(tf.shape(image)[0], dtype)
    orig_image_width = tf.cast(tf.shape(image)[1], dtype)

    mx = tf.maximum(orig_image_height, orig_image_width)
    mx_int = tf.cast(mx, tf.int32)
    image = tf.image.pad_to_bounding_box(image, tf.cast((mx - orig_image_height) / 2, tf.int32), tf.cast((mx - orig_image_width) / 2, tf.int32), mx_int, mx_int)

    image = tf.image.resize_with_pad(image, image_size, image_size)
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, [-1])

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
