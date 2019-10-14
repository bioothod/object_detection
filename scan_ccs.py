import argparse
import cv2
import json
import os

import numpy as np

def scan_tags(input_dir, images={}, annotations={}, categories={}):
    for fn in os.listdir(input_dir):
        image_filename_full = os.path.join(input_dir, fn)

        if os.path.isdir(image_filename_full):
            images, annotations, categories = scan_tags(image_filename_full, images, annotations, categories)
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

            images[image_id] = {
                'id': image_id,
                'file_name': image_filename_full,
            }

            with open(js_fn, 'r') as fin:
                js = json.load(fin)
                texts = js['texts']

                for x in texts:
                    text = x['text']
                    symbols = x['symbols']

                    def reset_text():
                        text_xmin = img_width
                        text_xmax = 0
                        text_ymin = img_height
                        text_ymax = 0

                        return text_xmin, text_ymin, text_xmax, text_ymax

                    text_xmin, text_ymin, text_xmax, text_ymax = reset_text()

                    for sym_idx, symbol in enumerate(symbols):
                        smb = symbol['symbol']
                        polys = np.array(symbol['p11m']) * np.array([img_width, img_height])

                        cat_id = categories[smb]['id']

                        xmin = polys[:, 0].min()
                        xmax = polys[:, 0].max()
                        ymin = polys[:, 1].min()
                        ymax = polys[:, 1].max()

                        text_xmin = min(xmin, text_xmin)
                        text_xmax = max(xmax, text_xmax)
                        text_ymin = min(ymin, text_ymin)
                        text_ymax = max(ymax, text_ymax)

                        ann_id = len(annotations)

                        annotations[ann_id] = {
                            'category_id': cat_id,
                            'image_id': image_id,
                            'id': ann_id,
                            'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                        }

                        if (sym_idx + 1) % 4 == 0:
                            ann_id = len(annotations)

                            cat_id = categories['text']['id']
                            annotations[ann_id] = {
                                'category_id': cat_id,
                                'image_id': image_id,
                                'id': ann_id,
                                'bbox': [text_xmin, text_ymin, text_xmax-text_xmin, text_ymax-text_ymin],
                            }

                            text_xmin, text_ymin, text_xmax, text_ymax = reset_text()

    return images, annotations, categories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_file', required=True, type=str, help='Path to store annotations json file')
    parser.add_argument('--input_dir', required=True, type=str, help='Image data directory')
    FLAGS = parser.parse_args()

    categories = {
            '0': {'name': '0', 'id': 0},
            '1': {'name': '1', 'id': 1},
            '2': {'name': '2', 'id': 2},
            '3': {'name': '3', 'id': 3},
            '4': {'name': '4', 'id': 4},
            '5': {'name': '5', 'id': 5},
            '6': {'name': '6', 'id': 6},
            '7': {'name': '7', 'id': 7},
            '8': {'name': '8', 'id': 8},
            '9': {'name': '9', 'id': 9},
            'text': {'name': 'text', 'id': 10},
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
