import argparse
import json
import os
import random

from collections import defaultdict

def scan_tags(input_dir, images={}, annotations={}, image_annotations={}, categories={}, image_categories={}):
    for fn in os.listdir(input_dir):
        image_filename_full = os.path.join(input_dir, fn)

        #if len(annotations) > 1000:
        #    break

        if os.path.isdir(image_filename_full):
            images, annotations, image_annotations, categories, image_categories = scan_tags(image_filename_full, images, annotations, image_annotations, categories, image_categories)
            continue

        image_extensions = ['.jpg', '.jpeg', '.png']
        ext = os.path.splitext(image_filename_full)
        if ext[1].lower() in image_extensions:
            image_name_full = ext[0]
            image_name = os.path.basename(image_name_full)
            user_id, photo_id = image_name.split('_')[:2]
            image_id = int(user_id) * int(photo_id)

            js_fn = '{}.json'.format(image_name_full)

            images[image_id] = {
                'id': image_id,
                'file_name': image_filename_full,
            }

            with open(js_fn, 'r') as fin:
                js = json.load(fin)

                js_categories = js['categories']

                for x in js_categories:
                    name = x['name']
                    boxes = x['rectangles']

                    if not boxes:
                        if name not in image_categories:
                            cat_id = len(image_categories)
                            image_categories[name] = {
                                'name': name,
                                'id': cat_id,
                            }
                            print('added new whole image category: {}/{}, cats: {}'.format(name, cat_id, len(image_categories)))

                        cat_id = image_categories[name]['id']

                        ann_id = len(image_annotations)

                        image_annotations[ann_id] = {
                            'category_id': cat_id,
                            'image_id': image_id,
                            'id': ann_id,
                        }
                    else:
                        if name not in categories:
                            cat_id = len(categories)
                            categories[name] = {
                                'name': name,
                                'id': cat_id,
                            }

                            print('added new category: {}/{}, cats: {}'.format(name, cat_id, len(categories)))

                        cat_id = categories[name]['id']

                        for box in boxes:
                            xmin = float(box['x1'])
                            xmax = float(box['x2'])
                            ymin = float(box['y1'])
                            ymax = float(box['y2'])

                            ann_id = len(annotations)

                            annotations[ann_id] = {
                                'category_id': cat_id,
                                'image_id': image_id,
                                'id': ann_id,
                                'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                            }

    print('input_dir: {}, images: {}, annotations: {}, whole_image_annotations: {}, categories: {}, whole_image_categories: {}'.format(
        input_dir, len(images),
        len(annotations), len(image_annotations),
        len(categories), len(image_categories)))
    return images, annotations, image_annotations, categories, image_categories

categories = {
        'interface': {'name': 'interface', 'id': 0},
        'contact_details': {'name': 'contact_details', 'id': 1},
        'underwear': {'name': 'underwear', 'id': 2},
        'erotic': {'name': 'erotic', 'id': 3},
        'element_overlay': {'name': 'element_overlay', 'id': 4},
        'weapon': {'name': 'weapon', 'id': 5},
        'text_overlay': {'name': 'text_overlay', 'id': 6},
        'child': {'name': 'child', 'id': 7},
        'sex': {'name': 'sex', 'id': 8},
        'violence': {'name': 'violence', 'id': 9},
        'blood': {'name': 'blood', 'id': 10},
        'drugs': {'name': 'drugs', 'id': 11},
        'male_face': {'name': 'drugs', 'id': 12},
        'person': {'name': 'drugs', 'id': 13},
        'female_face': {'name': 'drugs', 'id': 14},
        'good_element_overlay': {'name': 'good_element_overlay', 'id': 15},
        'trash_element': {'name': 'trash_element', 'id': 16},
        'hate_symbol': {'name': 'hate_symbol', 'id': 17},
        'normal_text': {'name': 'normal_text', 'id': 18},
        'naked_child': {'name': 'naked_child', 'id': 19},
        'qr_code': {'name': 'qr_code', 'id': 20},
}

image_categories = {
        'mirror_selfie': {'name': 'mirror_selfie', 'id': 0},
        'one_color_photo': {'name': 'one_color_photo', 'id': 1},
        'photo_of_photo': {'name': 'photo_of_photo', 'id': 2},
}

"""
input dir: /shared2/bo_datasets/photo_tagging/sort
annotations file: /shared2/object_detection/datasets/badoo/annotations/new/ann.json
date: 12 May 2020

object cat_id: 0, name: interface, annotations: 42539
object cat_id: 1, name: contact_details, annotations: 16212
object cat_id: 2, name: underwear, annotations: 17641
object cat_id: 3, name: erotic, annotations: 22543
object cat_id: 4, name: element_overlay, annotations: 42804
object cat_id: 5, name: weapon, annotations: 36602
object cat_id: 6, name: text_overlay, annotations: 43260
object cat_id: 7, name: child, annotations: 61979
object cat_id: 8, name: sex, annotations: 2427
object cat_id: 9, name: violence, annotations: 1086
object cat_id: 10, name: blood, annotations: 981
object cat_id: 11, name: drugs, annotations: 1138
object cat_id: 12, name: male_face, annotations: 149747
object cat_id: 13, name: person, annotations: 232891
object cat_id: 14, name: female_face, annotations: 103384
object cat_id: 15, name: good_element_overlay, annotations: 24448
object cat_id: 16, name: trash_element, annotations: 6007
object cat_id: 17, name: hate_symbol, annotations: 3752
object cat_id: 18, name: normal_text, annotations: 16616
object cat_id: 19, name: naked_child, annotations: 2166
object cat_id: 20, name: qr_code, annotations: 214
whole image cat_id: 0, name: mirror_selfie, annotations: 14171
whole image cat_id: 1, name: one_color_photo, annotations: 4988
whole image cat_id: 2, name: photo_of_photo, annotations: 3326
images: 287837, tags: 21
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_file', type=str, help='Path to store annotations json file')
    parser.add_argument('--input_dir', required=True, type=str, help='Image data directory')
    FLAGS = parser.parse_args()


    images, annotations, image_annotations, categories, image_categories = scan_tags(FLAGS.input_dir, categories=categories, image_categories=image_categories)

    def print_stats(prefix, annotations, categories):
        category_annotations = defaultdict(int)
        for ann_id, ann in annotations.items():
            cat_id = ann['category_id']
            category_annotations[cat_id] += 1

        for cat_name, cat in categories.items():
            cat_id = cat['id']
            count = category_annotations[cat_id]
            print('{} cat_id: {}, name: {}, annotations: {}'.format(prefix, cat_id, cat_name, count))

    print_stats('object', annotations, categories)
    print_stats('whole image', image_annotations, image_categories)

    images = list(images.values())
    annotations = list(annotations.values())
    image_annotations = list(image_annotations.values())
    categories = list(categories.values())
    image_categories = list(image_categories.values())


    output = {
        'annotations': annotations,
        'image_annotations': image_annotations,
        'categories': categories,
        'image_categories': image_categories,
        'images': images,
    }

    print('images: {}, tags: {}'.format(len(images), len(categories)))
    if FLAGS.annotations_file:
        with open(FLAGS.annotations_file, 'w') as f:
            json.dump(output, f, indent=2)
