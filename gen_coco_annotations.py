import argparse
import logging
import time

import numpy as np

import coco

logger = logging.getLogger('generate')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--coco_annotations', type=str, required=True, help='Path to MS COCO dataset: annotations json file')
parser.add_argument('--coco_data_dir', type=str, required=True, help='Path to MS COCO dataset: image directory')
parser.add_argument('--output', type=str, required=True, help='Output annotation file')
FLAGS = parser.parse_args()

def main():
    base = coco.create_coco_iterable(FLAGS.coco_annotations, FLAGS.coco_data_dir, logger)

    num_images = len(base)
    num_classes = base.num_classes()
    cat_names = base.cat_names()

    processed = 0
    bboxes = 0

    start_time = time.time()
    with open(FLAGS.output, 'w') as fout:
        for idx in range(num_images):
            try:
                filename, image_id, image, true_bboxes, true_labels = base.process_image(idx, None, return_orig_format=True)
            except coco.ProcessingError as e:
                logger.error('idx: {}, error: {}'.format(idx, e))
                continue

            processed += 1
            if processed % 1000 == 0:
                logger.info('{}%: processed: {}/{}, speed: {:.1f} ms/image'.format(
                    int(processed / num_images * 100), processed, num_images,
                    (time.time() - start_time) / processed * 1000))

            output_str = filename + ' '
            for bbox, label in zip(true_bboxes, true_labels):
                label -= 1

                cx, cy, h, w = np.split(bbox, 4)

                cx = np.squeeze(cx)
                cy = np.squeeze(cy)
                w = np.squeeze(h)
                h = np.squeeze(w)

                xmin = cx - w/2
                xmax = cx + w/2
                ymin = cy - h/2
                ymax = cy + h/2

                output_str += '{},{},{},{},{} '.format(xmin, ymin, xmax, ymax, label)

            fout.write(output_str + '\n')

        logger.info('{}: processed: {}/{}, speed: {:.1f} ms/image'.format(
            int(processed / num_images * 100), processed, num_images,
            (time.time() - start_time) / processed * 1000))

if __name__ == '__main__':
    main()
