import logging

from functools import partial
from typing import Tuple, List, Union

import tensorflow as tf

logger = logging.getLogger('detection')

def scale_boxes(boxes: tf.Tensor,
                from_size: Tuple[int, int],
                to_size: Tuple[int, int]) -> tf.Tensor:
    """
    Scale boxes to a new image size.
    Converts boxes generated in an image with `from_size` dimensions
    to boxes that fit inside an image with `to_size` dimensions

    Parameters
    ----------
    boxes: tf.Tensor
        Boxes to be scale. Boxes must be declared in
        [xmin, ymin, xmax, ymax] format
    from_size: Tuple[int, int], (height, width)
        Dimensions of the image where the boxes are decalred
    to_size: Tuple[int, int]

    Returns
    -------
    tf.Tensor
        Scaled boxes
    """
    ratio_w = from_size[1] / to_size[1]
    ratio_h = from_size[0] / to_size[0]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 *= ratio_w
    x2 *= ratio_w
    y1 *= ratio_h
    y2 *= ratio_h

    return tf.concat([x1, y1, x2, y2], axis=1)


def normalize_bndboxes(boxes: tf.Tensor,
                       image_size: Tuple[int, int]) -> tf.Tensor:
    """
    Normalizes boxes so they can be image size agnostic

    Parameters
    ----------
    boxes: tf.Tensor of shape [N_BOXES, 4]
        Boxes to be normalize. Boxes must be declared in
        [xmin, ymin, xmax, ymax] format
    image_size: Tuple[int, int], (height, width)
        Dimensions of the image where the boxes are decalred

    Returns
    -------
    tf.Tensor of shape [N_BOXES, 4]
        Normalized boxes with values in range [0, 1]
    """
    h, w = image_size
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 /= (w - 1)
    x2 /= (w - 1)
    y1 /= (h - 1)
    y2 /= (h - 1)
    return tf.concat([x1, y1, x2, y2], axis=1)


def regress_bndboxes(boxes: tf.Tensor,
                     regressors: tf.Tensor) -> tf.Tensor:
    """
    Apply scale invariant regression to boxes.

    Parameters
    ----------
    boxes: tf.Tensor of shape [BATCH, N_BOXES, 4]
        Boxes to apply the regressors
    regressors: tf.Tensor of shape [BATCH, N_BOXES, 4]
        Scale invariant regressions

    Returns
    -------
    tf.Tensor
        Regressed boxes
    """
    mean = tf.constant([0., 0., 0., 0.], dtype=boxes.dtype)
    std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=boxes.dtype)

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (regressors[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (regressors[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (regressors[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (regressors[:, :, 3] * std[3] + mean[3]) * height

    return tf.stack([x1, y1, x2, y2], axis=2)


def clip_boxes(boxes: tf.Tensor,
               im_size: Tuple[int, int]) -> tf.Tensor:
    # TODO: Document this
    h, w = im_size

    h = tf.cast(h - 1, boxes.dtype)
    w = tf.cast(w - 1, boxes.dtype)

    x1 = tf.clip_by_value(boxes[:, :, 0], 0., w)
    y1 = tf.clip_by_value(boxes[:, :, 1], 0., h)
    x2 = tf.clip_by_value(boxes[:, :, 2], 0., w)
    y2 = tf.clip_by_value(boxes[:, :, 3], 0., h)

    return tf.stack([x1, y1, x2, y2], axis=2)

@tf.function
def nms(bboxes: tf.Tensor,
        class_scores: tf.Tensor,
        score_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        max_ret: int = 100,
        ) -> tf.Tensor:

    """
    Parameters
    ----------
    bboxes: tf.Tensor of shape [BATCH, N, 4]

    class_scores: tf.Tensor of shape [BATCH, N, NUM_CLASSES]

    score_threshold: float, default 0.1
        Classification score to keep the box

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]
        The list len is equal to batch size.
        list[0] contains the bboxes and corresponding label of the first element
        of the batch
        bboxes List[tf.Tensor of shape [N, 4]]
        labels: List[tf.Tensor of shape [N]]
        scores: List[tf.Tensor of shape [N]]
    """

    batch_size = tf.shape(bboxes)[0]
    num_classes = tf.shape(class_scores)[-1]

    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=-1)
    bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
    bboxes = tf.reshape(bboxes, [batch_size, -1, 4])

    labels = tf.argmax(class_scores, -1, output_type=tf.int32)
    scores = tf.math.reduce_max(class_scores, -1)

    cond_fn = lambda c, *args: c < num_classes

    @tf.function
    def body(c, written, c_bboxes, c_scores, c_labels, sampled_bboxes, sampled_scores, sampled_labels):
        class_index = tf.where(tf.equal(sampled_labels, c))
        selected_scores = tf.gather_nd(sampled_scores, class_index)
        selected_bboxes = tf.gather_nd(sampled_bboxes, class_index)

        #logger.info('sampled_bboxes: {}, selected_bboxes: {}, sampled_scores: {}, selected_scores: {}'.format(
        #    sampled_bboxes.shape, selected_bboxes.shape, sampled_scores.shape, selected_scores.shape))

        indices = tf.image.non_max_suppression(
                selected_bboxes,
                selected_scores,
                max_output_size=max_ret,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold)

        num = tf.shape(indices)[0]
        if num > 0:
            best_bboxes = tf.gather(selected_bboxes, indices)
            best_scores = tf.gather(selected_scores, indices)
            best_labels = tf.ones_like(best_scores, dtype=tf.int32) * c

            #tf.print('c:', c, ', written:', written, ', indices:', tf.shape(indices), ', best_bboxes:', best_bboxes[:3], ', best_scores:', best_scores)

            c_bboxes = c_bboxes.write(written, best_bboxes)
            c_scores = c_scores.write(written, best_scores)
            c_labels = c_labels.write(written, best_labels)
            written += 1

        return c+1, written, c_bboxes, c_scores, c_labels

    @tf.function
    def batch_body(batch_idx):
        bboxes_for_image = bboxes[batch_idx]
        scores_for_image = scores[batch_idx]
        labels_for_image = labels[batch_idx]

        c = tf.constant(0, dtype=tf.int32)
        written = tf.constant(0, dtype=tf.int32)

        batch_bboxes = tf.TensorArray(bboxes.dtype, size=0, dynamic_size=True, infer_shape=False)
        batch_scores = tf.TensorArray(scores.dtype, size=0, dynamic_size=True, infer_shape=False)
        batch_labels = tf.TensorArray(labels.dtype, size=0, dynamic_size=True, infer_shape=False)

        non_background_index = tf.where(scores_for_image > score_threshold)
        non_background_index = tf.squeeze(non_background_index, 1)

        sampled_bboxes = tf.gather(bboxes_for_image, non_background_index)
        sampled_scores = tf.gather(scores_for_image, non_background_index)
        sampled_labels = tf.gather(labels_for_image, non_background_index)

        body_fn = partial(body, sampled_bboxes=sampled_bboxes, sampled_scores=sampled_scores, sampled_labels=sampled_labels)

        _, _, batch_bboxes, batch_scores, batch_labels = tf.while_loop(
                cond_fn, body_fn,
                parallel_iterations=1,
                back_prop=False,
                loop_vars=[c, written, batch_bboxes, batch_scores, batch_labels])

        if batch_bboxes.size() > 0:
            batch_bboxes = batch_bboxes.concat()
            batch_scores = batch_scores.concat()
            batch_labels = batch_labels.concat()

            _, best_index = tf.math.top_k(batch_scores, tf.minimum(max_ret, tf.shape(batch_scores)[0]), sorted=True)

            best_bboxes = tf.gather(batch_bboxes, best_index)
            best_scores = tf.gather(batch_scores, best_index)
            best_labels = tf.gather(batch_labels, best_index)

            #tf.print('batch_idx:', batch_idx, ', batch_bboxes:', tf.shape(batch_bboxes), ', best_bboxes:', tf.shape(best_bboxes), ', batch_scores:', tf.shape(batch_scores))

            y1, x1, y2, x2 = tf.split(best_bboxes, 4, axis=-1)
            best_bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
            best_bboxes = tf.reshape(best_bboxes, [-1, 4])

            to_add = tf.maximum(max_ret - tf.shape(best_scores)[0], 0)
            best_bboxes = tf.pad(best_bboxes, [[0, to_add], [0, 0]] , 'CONSTANT', constant_values=0)
            best_scores = tf.pad(best_scores, [[0, to_add]], 'CONSTANT', constant_values=0)
            best_labels = tf.pad(best_labels, [[0, to_add]], 'CONSTANT', constant_values=0)
        else:
            best_bboxes = tf.zeros((max_ret, 4), dtype=bboxes.dtype)
            best_scores = tf.zeros((max_ret,), dtype=scores.dtype)
            best_labels = tf.zeros((max_ret,), dtype=labels.dtype)

        return best_bboxes, best_scores, best_labels

    return tf.map_fn(batch_body, tf.range(batch_size),
            parallel_iterations=1,
            back_prop=False,
            infer_shape=False,
            dtype=(bboxes.dtype, scores.dtype, labels.dtype))

def bbox_overlap(bboxes, gt_bboxes):
    """
    Calculates the overlap between proposal and ground truth bboxes.
    Some `gt_bboxes` may have been padded. The returned `iou` tensor for these
    bboxes will be -1.

    Parameters
    ----------
    bboxes: tf.Tensor with a shape of [batch_size, N, 4].
        N is the number of proposals before groundtruth assignment. The
        last dimension is the pixel coordinates in [xmin, ymin, xmax, ymax] form.
    gt_bboxes: tf.Tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4].
        This tensor might have paddings with a negative value.

    Returns
    -------
    tf.FloatTensor
        A tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
    """
    bb_x_min, bb_y_min, bb_x_max, bb_y_max = tf.split(value=bboxes, num_or_size_splits=4, axis=2)
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = tf.split(value=gt_bboxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = (tf.math.maximum(i_xmax - i_xmin, 0) *
              tf.math.maximum(i_ymax - i_ymin, 0))

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)

    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth bboxes.
    gt_invalid_mask = tf.less(tf.reduce_max(gt_bboxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(tf.zeros_like(bb_x_min, dtype=tf.bool), tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou

