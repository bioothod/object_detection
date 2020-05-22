import logging
import typing

from functools import partial

import tensorflow as tf

logger = logging.getLogger('detection')

@tf.function
def non_max_suppression(coords, scores, max_ret, iou_threshold):
    ymin, xmin, ymax, xmax = tf.split(coords, num_or_size_splits=4, axis=1)
    ymax = tf.squeeze(ymax, 1)
    xmax = tf.squeeze(xmax, 1)
    ymin = tf.squeeze(ymin, 1)
    xmin = tf.squeeze(xmin, 1)

    area = (xmax - xmin) * (ymax - ymin)

    idxs = tf.argsort(scores, direction='ASCENDING', stable=False)

    max_idx = tf.minimum(tf.shape(idxs)[0], max_ret)

    pick = tf.TensorArray(tf.int32, size=max_idx)
    written = 0

    for idx in tf.range(max_idx):
        last_idx = tf.shape(idxs)[0] - 1
        if idx >= last_idx:
            break

        #-----------------------------------------------------------------------
        # Grab the last index (ie. the most confident detection), remove it from
        # the list of indices to process, and put it on the list of picks
        #-----------------------------------------------------------------------
        i = idxs[last_idx]
        idxs = idxs[:last_idx]

        pick = pick.write(idx, i)
        written += 1
        suppress = []

        xmin_idx = tf.gather(xmin, idxs)
        xmax_idx = tf.gather(xmax, idxs)
        ymin_idx = tf.gather(ymin, idxs)
        ymax_idx = tf.gather(ymax, idxs)

        xmin_i = tf.gather(xmin, i)
        xmax_i = tf.gather(xmax, i)
        ymin_i = tf.gather(ymin, i)
        ymax_i = tf.gather(ymax, i)

        #-----------------------------------------------------------------------
        # Figure out the intersection with the remaining windows
        #-----------------------------------------------------------------------
        xxmin = tf.maximum(xmin_i, xmin_idx)
        xxmax = tf.minimum(xmax_i, xmax_idx)
        yymin = tf.maximum(ymin_i, ymin_idx)
        yymax = tf.minimum(ymax_i, ymax_idx)

        w = tf.maximum(0., xxmax-xxmin+1)
        h = tf.maximum(0., yymax-yymin+1)
        intersection = w*h

        #-----------------------------------------------------------------------
        # Compute IOU and suppress indices with IOU higher than a threshold
        #-----------------------------------------------------------------------
        area_i = tf.gather(area, i)
        area_idx = tf.gather(area, idxs)
        union = area_i + area_idx - intersection
        iou = intersection/union
        nonoverlap_index = tf.where(iou <= iou_threshold)
        nonoverlap_index = tf.squeeze(nonoverlap_index, 1)

        idxs = tf.gather(idxs, nonoverlap_index)

    pick = pick.stack()[:written]
    return pick
    #return tf.gather(coords, pick), tf.gather(scores, pick)

def per_image_supression(logits, image_size, num_classes, min_obj_score, min_score, min_size, max_ret, iou_threshold):
    coords, scores, labels, objectness = logits

    non_background_index = tf.where(tf.logical_and(
                                        tf.greater(objectness, min_obj_score),
                                        tf.greater(scores, min_score)))

    non_background_index = tf.squeeze(non_background_index, 1)

    sampled_coords = tf.gather(coords, non_background_index)
    sampled_scores = tf.gather(scores, non_background_index)
    sampled_labels = tf.gather(labels, non_background_index)
    sampled_objs = tf.gather(objectness, non_background_index)

    image_size = tf.cast(image_size, coords.dtype)

    ret_coords_yx, ret_scores, ret_cat_ids, ret_objs = [], [], [], []
    for cat_id in range(0, num_classes):
        class_index = tf.where(tf.equal(sampled_labels, cat_id))
        #tf.print('class_index:', class_index)

        selected_scores = tf.gather_nd(sampled_scores, class_index)
        #logger.info('class_index: {}, sampled_labels: {}, sampled_scores: {}, selected_scores: {}'.format(class_index.shape, sampled_labels.shape, sampled_scores.shape, selected_scores.shape))

        selected_coords = tf.gather_nd(sampled_coords, class_index)
        selected_objs = tf.gather_nd(sampled_objs, class_index)

        #tf.print('selected_scores:', selected_scores, 'selected_coords:', selected_coords)

        cx, cy, w, h = tf.split(selected_coords, num_or_size_splits=4, axis=1)
        cx = tf.squeeze(cx, 1)
        cy = tf.squeeze(cy, 1)
        w = tf.squeeze(w, 1)
        h = tf.squeeze(h, 1)

        small_cond = tf.logical_and(h >= min_size, w >= min_size)
        large_cond = tf.logical_and(h < image_size*2, w < image_size*2)
        index = tf.where(tf.logical_and(small_cond, large_cond))
        selected_scores = tf.gather(selected_scores, index)
        selected_scores = tf.squeeze(selected_scores, 1)
        selected_coords = tf.gather(selected_coords, index)
        selected_coords = tf.squeeze(selected_coords, 1)
        selected_objs = tf.gather(selected_objs, index)
        selected_objs = tf.squeeze(selected_objs, 1)

        cx, cy, w, h = tf.split(selected_coords, num_or_size_splits=4, axis=1)
        cx = tf.squeeze(cx, 1)
        cy = tf.squeeze(cy, 1)
        w = tf.squeeze(w, 1)
        h = tf.squeeze(h, 1)

        #tf.print('cx:', cx, 'cy:', cy, 'w:', w, 'h:', h)

        xmin = cx - w/2
        xmax = cx + w/2
        ymin = cy - h/2
        ymax = cy + h/2

        xmin = tf.maximum(0., xmin)
        ymin = tf.maximum(0., ymin)
        xmax = tf.minimum(image_size, xmax)
        ymax = tf.minimum(image_size, ymax)

        coords_yx = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        #scores_to_sort = selected_scores * selected_objs
        scores_to_sort = selected_objs
        if True:
            selected_indexes = tf.image.non_max_suppression(coords_yx, scores_to_sort, max_ret, iou_threshold=iou_threshold)
        else:
            selected_indexes = non_max_suppression(coords_yx, scores_to_sort, max_ret, iou_threshold=iou_threshold)

        #logger.info('selected_indexes: {}, selected_coords: {}, selected_scores: {}'.format(selected_indexes, selected_coords, selected_scores))
        selected_coords_yx = tf.gather(coords_yx, selected_indexes)
        selected_scores = tf.gather(selected_scores, selected_indexes)
        selected_objs = tf.gather(selected_objs, selected_indexes)

        ret_coords_yx.append(selected_coords_yx)
        ret_scores.append(selected_scores)
        ret_objs.append(selected_objs)

        num = tf.shape(selected_scores)[0]
        tile = tf.tile([cat_id], [num])
        ret_cat_ids.append(tile)

    ret_coords_yx = tf.concat(ret_coords_yx, 0)
    ret_scores = tf.concat(ret_scores, 0)
    ret_objs = tf.concat(ret_objs, 0)
    ret_cat_ids = tf.concat(ret_cat_ids, 0)

    #logger.info('ret_coords: {}, ret_scores: {}, ret_cat_ids: {}'.format(ret_coords, ret_scores, ret_cat_ids))

    #scores_to_sort = ret_scores * ret_objs
    scores_to_sort = ret_objs
    _, best_index = tf.math.top_k(scores_to_sort, tf.minimum(max_ret, tf.shape(ret_scores)[0]), sorted=True)

    best_scores = tf.gather(ret_scores, best_index)
    best_coords_yx = tf.gather(ret_coords_yx, best_index)
    best_objs = tf.gather(ret_objs, best_index)
    best_cat_ids = tf.gather(ret_cat_ids, best_index)

    y0, x0, y1, x1 = tf.split(best_coords_yx, 4, axis=1)
    best_coords = tf.concat([x0, y0, x1, y1], 1)

    #logger.info('best_coords: {}, best_scores: {}, best_cat_ids: {}'.format(best_coords, best_scores, best_cat_ids))

    to_add = tf.maximum(max_ret - tf.shape(best_scores)[0], 0)
    best_coords = tf.pad(best_coords, [[0, to_add], [0, 0]] , 'CONSTANT', constant_values=0)
    best_scores = tf.pad(best_scores, [[0, to_add]], 'CONSTANT', constant_values=0)
    best_objs = tf.pad(best_objs, [[0, to_add]], 'CONSTANT', constant_values=0)
    best_cat_ids = tf.pad(best_cat_ids, [[0, to_add]], 'CONSTANT', constant_values=0)


    #logger.info('ret_coords: {}, ret_scores: {}, ret_cat_ids: {}, best_index: {}, best_scores: {}, best_coords: {}, best_cat_ids: {}'.format(
    #    ret_coords, ret_scores, ret_cat_ids, best_index, best_scores, best_coords, best_cat_ids))

    return best_coords, best_scores, best_objs, best_cat_ids

def make_predictions(model: tf.keras.Model,
                     images: tf.Tensor,
                     all_anchors: tf.Tensor,
                     all_grid_xy: tf.Tensor,
                     all_ratios: tf.Tensor,
                     min_obj_score: float = 0.7,
                     min_score: float = 0.8,
                     min_size: int = 16,
                     max_ret: int = 100,
                     iou_threshold: float = 0.45,
                     ) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        image_size = tf.shape(images)[1]

        pred_values = model(images, training=False)

        pred_objs = tf.math.sigmoid(pred_values[..., 4])

        pred_bboxes = pred_values[..., 0:4]
        pred_scores_all = tf.math.sigmoid(pred_values[..., 5:])
        pred_scores = tf.reduce_max(pred_scores_all, axis=-1)
        pred_labels = tf.argmax(pred_scores_all, axis=-1)

        num_classes = tf.shape(pred_scores_all)[-1].numpy()

        all_ratios = tf.expand_dims(all_ratios, -1)

        pred_xy = (tf.sigmoid(pred_bboxes[..., 0:2]) + all_grid_xy) * all_ratios
        pred_wh = tf.math.exp(pred_bboxes[..., 2:4]) * all_anchors[..., 2:4]

        pred_bboxes = tf.concat([pred_xy, pred_wh], axis=-1)

        return tf.map_fn(lambda out: per_image_supression(out, image_size, num_classes, min_obj_score, min_score, min_size, max_ret, iou_threshold),
                         (pred_bboxes, pred_scores, pred_labels, pred_objs),
                         parallel_iterations=16,
                         back_prop=False,
                         dtype=(tf.float32, tf.float32, tf.float32, tf.int32))

