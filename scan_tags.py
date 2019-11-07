import argparse
import json
import os
import random

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
            user_id, photo_id = image_name.split('_')[:2]
            image_id = int(user_id) * int(photo_id)

            js_fn = '{}.json'.format(image_name_full)

            images[image_id] = {
                'id': image_id,
                'file_name': image_filename_full,
            }

            with open(js_fn, 'r') as fin:
                js = json.load(fin)

                for x in js:
                    name = x['name']
                    boxes = x['rectangles']
                    if boxes is not None:
                        if name not in categories:
                            cat_id = len(categories)
                            categories[name] = {
                                'name': name,
                                'id': cat_id,
                            }

                            print('added new category: {}/{}, cats: {}'.format(name, cat_id, categories))

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

    return images, annotations, categories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_file', required=True, type=str, help='Path to store annotations json file')
    parser.add_argument('--input_dir', required=True, type=str, help='Image data directory')
    FLAGS = parser.parse_args()

    categories = {
            'interface': {'name': 'interface', 'id': 0},
            'contact_details': {'name': 'contact_details', 'id': 1},
            'face': {'name': 'face', 'id': 2},
            'underwear': {'name': 'underwear', 'id': 3},
            'erotic': {'name': 'erotic', 'id': 4},
            'element_overlay': {'name': 'element_overlay', 'id': 5},
            'weapon': {'name': 'weapon', 'id': 6},
            'text_overlay': {'name': 'text_overlay', 'id': 7},
            'child': {'name': 'child', 'id': 8},
            'hate_gesture': {'name': 'hate_gesture', 'id': 9},
            'sex': {'name': 'sex', 'id': 10},
            'violence': {'name': 'violence', 'id': 11},
            'blood': {'name': 'blood', 'id': 12},
            'drugs': {'name': 'drugs', 'id': 13},
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
