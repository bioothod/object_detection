from functools import partial
from typing import Tuple, List, Union

import tensorflow as tf


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
    boxes = tf.cast(boxes, tf.float32)
    regressors = tf.cast(regressors, tf.float32)

    mean = tf.constant([0., 0., 0., 0.], dtype=tf.float32)
    std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)

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

    h = tf.cast(h - 1, tf.float32)
    w = tf.cast(w - 1, tf.float32)

    x1 = tf.clip_by_value(boxes[:, :, 0], 0., w)
    y1 = tf.clip_by_value(boxes[:, :, 1], 0., h)
    x2 = tf.clip_by_value(boxes[:, :, 2], 0., w)
    y2 = tf.clip_by_value(boxes[:, :, 3], 0., h)

    return tf.stack([x1, y1, x2, y2], axis=2)

@tf.function
def nms(bboxes: tf.Tensor,
        class_scores: tf.Tensor,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
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

    # TF while loop variables
    cond_fn = lambda c, *args: c < num_classes

    bboxes = tf.cast(bboxes, tf.float32)
    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=-1)
    bboxes = tf.stack([y1, x1, y2, x2], axis=-1)
    bboxes = tf.reshape(bboxes, [batch_size, -1, 4])

    class_scores = tf.cast(class_scores, tf.float32)

    all_bboxes = []
    all_labels = []
    all_scores = []

    @tf.function
    def body(c, written, c_bboxes, c_scores, c_labels, batch_idx):
        nms_scores = tf.gather(class_scores[batch_idx], c, axis=-1)
        nms_scores = tf.reshape(nms_scores, [-1])

        bboxes_for_image = bboxes[batch_idx]

        indices = tf.image.non_max_suppression(
                bboxes_for_image,
                nms_scores,
                max_output_size=max_ret,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold)

        num = tf.shape(indices)[0]
        if num != 0:
            best_bboxes = tf.gather(bboxes_for_image, indices)
            best_scores = tf.gather(nms_scores, indices)
            best_labels = tf.ones([tf.shape(indices)[0]], dtype=tf.int32) * c

            c_bboxes = c_bboxes.write(written, best_bboxes)
            c_scores = c_scores.write(written, best_scores)
            c_labels = c_labels.write(written, best_labels)
            written += 1

        return c + 1, written, c_bboxes, c_scores, c_labels

    @tf.function
    def batch_body(batch_idx):
        body_fn = partial(body, batch_idx=batch_idx)
        # For each class, get the effective bboxes, labels and scores
        c = 0
        written = 0
        batch_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        batch_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        batch_labels = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)

        _, _, batch_bboxes, batch_scores, batch_labels = tf.while_loop(
                cond_fn, body_fn,
                parallel_iterations=32,
                back_prop=False,
                loop_vars=[c, written, batch_bboxes, batch_scores, batch_labels])

        if batch_bboxes.size() != 0:
            batch_bboxes = batch_bboxes.concat()
            batch_scores = batch_scores.concat()
            batch_labels = batch_labels.concat()

            _, best_index = tf.math.top_k(batch_scores, tf.minimum(max_ret, tf.shape(batch_scores)[0]), sorted=True)

            batch_bboxes = tf.gather(batch_bboxes, best_index)
            batch_scores = tf.gather(batch_scores, best_index)
            batch_labels = tf.gather(batch_labels, best_index)

            y1, x1, y2, x2 = tf.split(batch_bboxes, 4, axis=-1)
            batch_bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
            batch_bboxes = tf.reshape(batch_bboxes, [-1, 4])

            to_add = tf.maximum(max_ret - tf.shape(batch_scores)[0], 0)
            batch_bboxes = tf.pad(batch_bboxes, [[0, to_add], [0, 0]] , 'CONSTANT')
            batch_scores = tf.pad(batch_scores, [[0, to_add]], 'CONSTANT')
            batch_labels = tf.pad(batch_labels, [[0, to_add]], 'CONSTANT')
        else:
            batch_bboxes = tf.zeros([max_ret, 4], tf.float32)
            batch_scores = tf.zeros([max_ret], tf.float32)
            batch_labels = tf.zeros([max_ret], tf.int32)

        return batch_bboxes, batch_scores, batch_labels

    return tf.map_fn(batch_body, tf.range(batch_size),
            parallel_iterations=32,
            back_prop=False,
            infer_shape=False,
            dtype=(tf.float32, tf.float32, tf.int32))

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

