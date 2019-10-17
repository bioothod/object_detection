import argparse
import cv2
import json
import os

import numpy as np

def float2abs(coord, image_shape):
    return [coord[0] * image_shape[1], coord[1] * image_shape[0]]

def find_poly(obj, image_shape):
    poly_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

    poly = [float2abs(obj[name], image_shape) for name in poly_names]
    return poly

def scan_tags(input_dir, images={}, annotations={}, categories={}):
    for fn in os.listdir(input_dir):
        image_filename_full = os.path.join(input_dir, fn)

        if os.path.isdir(image_filename_full):
            images, annotations, categories = scan_tags(image_filename_full, images, annotations, categories)
            continue

        image_extensions = ['.jpg', '.jpeg', '.png']
        ext_split = os.path.splitext(image_filename_full)
        ext_lower = ext_split[1].lower()
        if ext_lower in image_extensions:
            image_name_full = ext_split[0]
            image_name = os.path.basename(image_name_full)

            exts = image_filename_full.lower().split(ext_lower)
            if len(exts) > 2:
                continue

            js_fn = '{}.json'.format(image_filename_full)
            if not os.path.exists(js_fn):
                continue

            image_id = int(image_name[3:])

            image = cv2.imread(image_filename_full)

            images[image_id] = {
                'id': image_id,
                'file_name': image_filename_full,
            }

            with open(js_fn, 'r') as fin:
                js = json.load(fin)

                card = js['card']
                poly_list = find_poly(card, image.shape)
                poly = np.array(poly_list, dtype=np.float32)

                margin = 0

                xmin = poly[:, 0].min() - margin
                xmax = poly[:, 0].max() + margin
                ymin = poly[:, 1].min() - margin
                ymax = poly[:, 1].max() + margin

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > image.shape[1]:
                    xmax = image.shape[1]
                if ymax > image.shape[0]:
                    ymax = image.shape[0]

                ann_id = len(annotations)
                cat_id = categories['card']['id']

                annotations[ann_id] = {
                    'category_id': cat_id,
                    'image_id': image_id,
                    'id': ann_id,
                    'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                    'keypoints': poly_list,
                }

    return images, annotations, categories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_file', required=True, type=str, help='Path to store annotations json file')
    parser.add_argument('--input_dir', required=True, type=str, help='Image data directory')
    FLAGS = parser.parse_args()

    categories = {
            'card': {'name': 'card', 'id': 0},
    }
    images, annotations, categories = scan_tags(FLAGS.input_dir, categories=categories)


    images = list(images.values())
    annotations = list(annotations.values())
    categories = list(categories.values())

    output = {
        'annotations': annotations,
        'categories': categories,
        'images': images,
    }

    print('images: {}, tags: {}'.format(len(images), len(categories)))
    with open(FLAGS.annotations_file, 'w') as f:
        json.dump(output, f)
