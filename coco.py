import json
import logging

from collections import defaultdict

logger = logging.getLogger('objdet')

class COCO:
    def __init__(self, path):
        self.load_instance_json(path)

    def load_instance_json(self, path):
        logging.info('loading instance json {}'.format(path))
        self.dataset = json.load(open(path, 'r'))

        logging.info('generating index from {}'.format(path))
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.img2anns = imgToAnns
        self.cat2mgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        num = 0
        for image_id, anns in self.img2anns.items():
            for ann in anns:
                bbox = ann['bbox']
                category_id = ann['category_id']
                category_name = self.cats[category_id]['name']
                logger.info('image: {}, bbox: {}, category: {}/{}'.format(image_id, bbox, category_name, category_id))

            num += 1
            if num > 2:
                break

        logger.info('annotations: {}, images: {}, categories: {}'.format(len(self.anns), len(self.imgs), len(self.cats)))
        category_names = [c['name'] for c in self.cats.values()]
        logger.info('categories: {}'.format(len(category_names)))

        self.get_images()

    def get_images(self):
        file_ids = []
        filenames = []
        for img in self.imgs.values():
            file_ids.append(img['id'])
            filenames.append(img['file_name'])

        logger.info(self.imgs[289343])
