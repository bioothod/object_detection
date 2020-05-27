import copy
import logging
import os
import time
import typing

import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO

from cocoeval import COCOeval

import bndbox
import image as image_draw

logger = logging.getLogger('detection')

def _COCO_result(image_id: int,
                 labels: tf.Tensor,
                 bboxes: tf.Tensor,
                 scores: tf.Tensor):

    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()
    scores = scores.numpy().tolist()

    return [dict(image_id=image_id, category_id=l, bbox=b, score=s)
            for l, b, s in zip(labels, coco_bboxes, scores)]


def _COCO_gt_annot(image_id: int,
                   annot_id: int,
                   image_shape: typing.Tuple[int, int],
                   labels: tf.Tensor,
                   bboxes: tf.Tensor):

    im_h, im_w = image_shape

    b_h = bboxes[:, 3] - bboxes[:, 1]
    b_w = bboxes[:, 2] - bboxes[:, 0]
    areas = tf.reshape(b_h * b_w, [-1])
    areas = areas.numpy().tolist()

    coco_bboxes = tf.stack([bboxes[:, 0], bboxes[:, 1], b_w, b_h])
    coco_bboxes = tf.transpose(coco_bboxes).numpy().tolist()

    labels = labels.numpy().tolist()

    image = dict(id=image_id, height=im_h, width=im_w)

    it = zip(coco_bboxes, areas, labels)
    annotations = [dict(id=id_, image_id=image_id, bbox=bbox, iscrowd=0, area=a, category_id=l)
            for id_, (bbox, a, l) in enumerate(it, start=annot_id)]

    return image, annotations


def evaluate(model: tf.keras.Model,
             dataset: tf.data.Dataset,
             class2idx: typing.Mapping[str, int],
             all_anchors: tf.Tensor,
             all_grid_xy: tf.Tensor,
             all_ratios: tf.Tensor,
             steps: int,
             data_dir: str,
             save_examples: int = -1,
             print_every: int = 10):

    gt_coco = dict(images=[], annotations=[])
    results_coco = []
    image_id = 1
    annot_id = 1

    all_ratios_ext = tf.expand_dims(all_ratios, -1)

    # Create COCO categories
    categories = [dict(supercategory='instance', id=i, name=n)
                  for n, i in class2idx.items()]
    gt_coco['categories'] = categories

    cat_names = {}
    for cname, cid in class2idx.items():
        cat_names[cid] = cname

    if save_examples > 0:
        data_dir = os.path.join(data_dir, 'evaluation')
        os.makedirs(data_dir, exist_ok=True)

    min_obj_score = 0.3
    min_score = 0.5
    iou_threshold = 0.3

    start_time = time.time()
    total_time = 0.
    num_images = 0
    i = 0

    for filenames, image_ids, images, true_values in dataset:
        inference_start = time.time()
        pred_bboxes, pred_scores, pred_objs, pred_cat_ids = bndbox.make_predictions(model, images,
                all_anchors, all_grid_xy, all_ratios,
                num_classes=len(categories),
                min_obj_score=min_obj_score, min_score=min_score, iou_threshold=iou_threshold)

        h, w = images[0].shape[1 : 3]

        # Iterate through images in batch, and for each one
        # create the ground truth coco annotation

        for batch_idx in range(pred_bboxes.shape[0]):
            val = true_values[batch_idx, ...]
            #tf.print('val:', tf.shape(val), ', all_grid_xy:', tf.shape(all_grid_xy), ', all_ratios_ext:', tf.shape(all_ratios_ext), ', all_anchors:', tf.shape(all_anchors))

            non_background_index = tf.where(val[..., 4] != 0)
            non_background_index = tf.squeeze(non_background_index, 1)
            val = tf.gather(val, non_background_index)
            #tf.print(filenames[batch_idx], ', non_background_index:', tf.shape(non_background_index), ', val:', tf.shape(val))

            true_bboxes = val[:, 0:4]
            true_labels = val[:, 5:]
            true_labels = tf.argmax(true_labels, axis=1)

            grid_xy = tf.gather(all_grid_xy, non_background_index)
            ratios = tf.gather(all_ratios_ext, non_background_index)
            anchors_wh = tf.gather(all_anchors, non_background_index)

            #tf.print('grid_xy:', tf.shape(grid_xy), ', ratios:', tf.shape(ratios), ', anchors_wh:', tf.shape(anchors_wh), ', true_bboxes:', tf.shape(true_bboxes))

            true_xy = (true_bboxes[:, 0:2] + grid_xy) * ratios
            true_wh = tf.math.exp(true_bboxes[:, 2:4]) * anchors_wh[:, 2:4]

            cx = true_xy[..., 0]
            cy = true_xy[..., 1]
            w = true_wh[..., 0]
            h = true_wh[..., 1]

            x0 = cx - w/2
            x1 = cx + w/2
            y0 = cy - h/2
            y1 = cy + h/2

            true_bboxes = tf.stack([x0, y0, x1, y1], axis=1)

            im_annot, annots = _COCO_gt_annot(image_id, annot_id, (h, w), true_labels, true_bboxes)
            gt_coco['annotations'].extend(annots)
            gt_coco['images'].append(im_annot)

            pred_bboxes_for_image = pred_bboxes[batch_idx, ...]
            pred_objs_for_image = pred_objs[batch_idx, ...]
            pred_scores_for_image = pred_scores[batch_idx, ...]
            pred_labels_for_image = pred_cat_ids[batch_idx, ...]

            good_idx = tf.where(pred_objs_for_image >= min_obj_score)
            good_idx = tf.squeeze(good_idx, 1)
            pred_bboxes_for_image = tf.gather(pred_bboxes_for_image, good_idx)
            pred_scores_for_image = tf.gather(pred_scores_for_image, good_idx)
            pred_labels_for_image = tf.gather(pred_labels_for_image, good_idx)

            #tf.print(filenames[batch_idx], ', pred_scores_for_image:', pred_scores_for_image, ', labels:', pred_labels_for_image)

            if tf.shape(pred_labels_for_image)[0] > 0:
                results = _COCO_result(image_id, pred_labels_for_image, pred_bboxes_for_image, pred_scores_for_image)
                results_coco.extend(results)

            if save_examples > 0:
                new_anns = []
                for bb, label in zip(true_bboxes, true_labels):
                    if label == -1:
                        continue

                    label = 0
                    new_anns.append((bb, None, label))

                for bb, label in zip(pred_bboxes_for_image, pred_labels_for_image):
                    if label == -1:
                        continue

                    label = 1
                    new_anns.append((bb, None, label))

                filename = filenames[batch_idx]
                filename = str(filename.numpy(), 'utf8')
                filename_base = os.path.basename(filename)
                filename_base = os.path.splitext(filename_base)[0]

                image = images[batch_idx]
                image = image.numpy()
                image = image * 128 + 128
                image = image.astype(np.uint8)

                dst = os.path.join(data_dir, filename_base) + '.png'
                image_draw.draw_im(image, new_anns, dst, {})

            annot_id += len(annots)
            image_id += 1
            num_images += 1

        total_time += time.time() - inference_start

        if i % print_every == 0:
            logger.info('validated steps: {}/{}, images: {}, perf: {:.1f} img/s, time_per_image: {:.1f} ms'.format(
                i, steps, num_images, num_images / total_time, total_time / num_images * 1000))

        i += 1
        if i == steps:
            break

    total_time = time.time() - start_time
    if num_images == 0:
        logger.info('there are no validation images, returning 0 from evaluation')
        return 0.

    logger.info('validated steps: {}, images: {}, perf: {:.1f}  img/s, time_per_image: {:.1f} ms'.format(
        i, num_images, num_images / total_time, total_time / num_images * 1000))

    # Convert custom annotations to COCO annots
    gtCOCO = COCO()
    gtCOCO.dataset = gt_coco
    gtCOCO.createIndex()

    resCOCO = COCO()
    resCOCO.dataset['images'] = [img for img in gt_coco['images']]
    resCOCO.dataset['categories'] = copy.deepcopy(gt_coco['categories'])

    for i, ann in enumerate(results_coco):
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if not 'segmentation' in ann:
            ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        ann['area'] = bb[2]*bb[3]
        ann['id'] = i + 1
        ann['iscrowd'] = 0

    resCOCO.dataset['annotations'] = results_coco
    resCOCO.createIndex()

    coco_eval = COCOeval(gtCOCO, resCOCO, 'bbox', logger=logger)
    coco_eval.params.imgIds = sorted(gtCOCO.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]
