import argparse
import cv2
import json
import logging
import os

import numpy as np
import tensorflow as tf

import tfrecord_writer

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def scan_tags(input_dir, writer):
    processed = 0
    logger.info('start scanning directory {}'.format(input_dir))

    for fn in os.listdir(input_dir):
        image_filename_full = os.path.join(input_dir, fn)

        if os.path.isdir(image_filename_full):
            scan_tags(image_filename_full, writer)
            continue

        image_extensions = ['.jpg', '.jpeg', '.png']
        ext = os.path.splitext(image_filename_full)
        if ext[1].lower() in image_extensions:
            image_name_full = ext[0]
            image_name = os.path.basename(image_name_full)

            js_fn = '{}.json'.format(image_filename_full)
            if not os.path.exists(js_fn):
                continue

            image_id = int(image_name[3:])

            image = cv2.imread(image_filename_full)
            img_height, img_width, _ = image.shape

            with open(image_filename_full, 'rb') as fin:
                image_data = fin.read()

            with open(js_fn, 'r') as fin:
                js = json.load(fin)
                texts = js['texts']

                char_poly = []
                word_poly = []
                word_bboxes = []
                text_strings = []
                chars = ''

                def update(text_poly, word):
                    if len(text_poly) != 0:
                        s = text_poly[0]
                        e = text_poly[-1]
                        wp = [s[0], e[1], e[2], s[3]]
                        word_poly.append(wp)
                        text_strings.append(word)

                        wp = np.array(wp, dtype=np.float32)
                        xmin = wp[0::2].min()
                        xmax = wp[0::2].max()
                        ymin = wp[1::2].min()
                        ymax = wp[1::2].max()

                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        h = ymax - ymin
                        w = xmax - xmin
                        text_bbox = np.array([cx, cy, h, w], dtype=np.float32)
                        word_bboxes.append(text_bbox)

                    return [], ''

                for x in texts:
                    text = x['text']
                    symbols = x['symbols']

                    text_poly = []
                    word = ''

                    for symbol in symbols:
                        smb = symbol['symbol']
                        poly = np.array(symbol['p11m']) * np.array([img_width, img_height])

                        if smb.startswith('_'):
                            smb = smb[1]

                        #logger.info('text_poly: {}, smb: "{}", word: "{}", count: {}'.format(len(text_poly), smb, word, word.count('/')))
                        if smb == ' ':
                            text_poly, word = update(text_poly, word)
                            continue

                        text_poly.append(poly)
                        char_poly.append(poly)

                        chars += smb
                        word += smb

                        if len(text_poly) == 4 + word.count('/'):
                            text_poly, word = update(text_poly, word)

                    text_poly, word = update(text_poly, word)


                texts = '<SEP>'.join(text_strings)

                char_poly = np.array(char_poly, dtype=np.float64)
                word_poly = np.array(word_poly, dtype=np.float32)
                word_bboxes = np.array(word_bboxes, dtype=np.float32)

                #logger.info('{}: word_poly: {}, words: {}, char_poly: {}, chars: {}'.format(image_id, word_poly.shape, len(text_strings), char_poly.shape, len(chars)))
                #exit(0)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image_data),
                    'filename': _bytes_feature(bytes(image_filename_full, 'UTF-8')),
                    'char_poly': _bytes_feature(char_poly.tobytes()),
                    'word_poly': _bytes_feature(word_poly.tobytes()),
                    'text': _bytes_feature(bytes(texts, 'UTF-8')),
                    'text_concat': _bytes_feature(bytes(chars, 'UTF-8')),
                    'true_labels': _bytes_feature(bytes(texts, 'UTF-8')),
                    'true_bboxes': _bytes_feature(word_bboxes.tobytes()),
                    }))

                data = bytes(example.SerializeToString())
                writer.write(data)

                processed += 1
                if processed % 1000 == 0:
                    logger.info('{}: written {}'.format(input_dir, processed))

    return processed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, action='append', type=str, help='Image data directory')
    parser.add_argument('--output_dir', required=True, type=str, help='Tfrecord output dir')
    parser.add_argument('--num_images_per_tfrecord', default=10000, type=int, help='Maximum number of images per tfrecord')
    FLAGS = parser.parse_args()

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    writer = tfrecord_writer.tf_records_writer('{}/tfrecord'.format(FLAGS.output_dir), 0, FLAGS.num_images_per_tfrecord)

    processed = 0
    for input_dir in FLAGS.input_dir:
        written = scan_tags(input_dir, writer)
        processed += written

        logger.info('{}: written {}, total written: {}'.format(input_dir, written, processed))

    writer.close()
