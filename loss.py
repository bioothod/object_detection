import logging

import tensorflow as tf

logger = logging.getLogger('detection')

class Metric:
    def __init__(self, **kwargs):
        self.total_loss = tf.keras.metrics.Mean()

        self.char_dist_loss = tf.keras.metrics.Mean()
        self.word_dist_loss = tf.keras.metrics.Mean()

        self.char_obj_loss = tf.keras.metrics.Mean()
        self.word_obj_loss = tf.keras.metrics.Mean()

        self.letters_ce_loss = tf.keras.metrics.Mean()
        self.letters_accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.char_obj_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.word_obj_accuracy = tf.keras.metrics.BinaryAccuracy()

    def reset_states(self):
        self.total_loss.reset_states()

        self.char_dist_loss.reset_states()
        self.word_dist_loss.reset_states()

        self.char_obj_loss.reset_states()
        self.word_obj_loss.reset_states()

        self.letters_ce_loss.reset_states()
        self.letters_accuracy.reset_states()

        self.char_obj_accuracy.reset_states()
        self.word_obj_accuracy.reset_states()

    def str_result(self):
        return 'total_loss: {:.3e}, dist: {:.3e}/{:.3e}, obj: {:.3e}/{:.3e}, letters_ce: {:.3e}, letters_accuracy: {:.4f}, obj_accuracy: {:.4f}/{:.4f}'.format(
                self.total_loss.result(),
                self.char_dist_loss.result(),
                self.word_dist_loss.result(),

                self.char_obj_loss.result(),
                self.word_obj_loss.result(),

                self.letters_ce_loss.result(),
                self.letters_accuracy.result(),

                self.char_obj_accuracy.result(),
                self.word_obj_accuracy.result(),
                )

class LossMetricAggregator:
    def __init__(self,
                 dictionary_size,
                 grid_xy,
                 global_batch_size,
                 **kwargs):

        #super(YOLOLoss, self).__init__(**kwargs)

        self.dictionary_size = dictionary_size

        self.global_batch_size = global_batch_size

        self.grid_xy = grid_xy

        # [N, 2] -> [N, 1, 2]
        self.grid_xy = tf.expand_dims(self.grid_xy, 1)
        # [N, 1, 2] -> [N, 4, 2]
        self.grid_xy = tf.tile(self.grid_xy, [1, 4, 1])

        # added batch dimension
        self.grid_xy = tf.expand_dims(self.grid_xy, 0)

        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.letters_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.obj_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1, from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.train_metric = Metric(name='train_metric')
        self.eval_metric = Metric(name='eval_metric')

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def evaluation_result(self):
        m = self.eval_metric
        obj_acc = m.char_obj_accuracy.result() + m.word_obj_accuracy.result()
        dist = tf.math.exp(-m.char_dist_loss.result()) + tf.math.exp(-m.word_dist_loss.result())
        letters_acc = m.letters_accuracy.result()

        return obj_acc * 2 + dist + letters_acc

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def gen_ignore_mask(self, input_tuple, object_mask, true_bboxes):
        idx, pred_boxes_for_single_image = input_tuple
        valid_true_boxes = tf.boolean_mask(true_bboxes[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], tf.bool))
        iou = box_iou(pred_boxes_for_single_image, valid_true_boxes)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        #logger.info('pred_boxes_for_single_image: {}, valid_true_boxes: {}, iou: {}, best_iou: {}, ignore_mask_tmp: {}'.format(
        #    pred_boxes_for_single_image.shape, valid_true_boxes.shape, iou.shape, best_iou.shape, ignore_mask_tmp.shape))
        return ignore_mask_tmp

    def focal_loss(self, y_true, y_pred, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(y_true - y_pred), gamma)
        return focal_loss

    def loss(self, y_true, y_pred, training):
        char_boundary_start = 0
        word_boundary_start = self.dictionary_size + 1 + 4 * 2

        # predicted tensors
        pred_char = y_pred[..., char_boundary_start : char_boundary_start + word_boundary_start]
        pred_word = y_pred[..., word_boundary_start : ]

        pred_char_obj = pred_char[..., 0]
        pred_char_poly = pred_char[..., 1 : 9]
        pred_char_letters = pred_char[..., 10 :]

        pred_word_obj = pred_word[..., 0]
        pred_word_poly = pred_word[..., 1 : 9]

        # true tensors
        true_char = y_true[..., char_boundary_start : char_boundary_start + word_boundary_start]
        true_word = y_true[..., word_boundary_start : ]

        true_char_obj = true_char[..., 0]
        true_char_poly = true_char[..., 1 : 9]
        true_char_letters = true_char[..., 10 :]

        true_word_obj = true_word[..., 0]
        true_word_poly = true_word[..., 1 : 9]

        true_char_obj_whole = true_char_obj
        pred_char_obj_whole = pred_char_obj
        true_word_obj_whole = true_word_obj
        pred_word_obj_whole = pred_word_obj

        char_object_mask = tf.cast(true_char_obj, 'bool')
        word_object_mask = tf.cast(true_word_obj, 'bool')


        pred_char_obj = tf.boolean_mask(pred_char_obj, char_object_mask)
        pred_char_poly = tf.boolean_mask(pred_char_poly, char_object_mask)
        pred_char_letters = tf.boolean_mask(pred_char_letters, char_object_mask)

        pred_word_obj = tf.boolean_mask(pred_word_obj, word_object_mask)
        pred_word_poly = tf.boolean_mask(pred_word_poly, word_object_mask)


        true_char_obj = tf.boolean_mask(true_char_obj, char_object_mask)
        true_char_poly = tf.boolean_mask(true_char_poly, char_object_mask)
        true_char_letters = tf.boolean_mask(true_char_letters, char_object_mask)

        true_word_obj = tf.boolean_mask(true_word_obj, word_object_mask)
        true_word_poly = tf.boolean_mask(true_word_poly, word_object_mask)

        m = self.train_metric
        if not training:
            m = self.eval_metric


        # losses

        # distance loss
        char_dist_loss = self.mae(true_char_poly, pred_char_poly)
        word_dist_loss = self.mae(true_word_poly, pred_word_poly)

        m.char_dist_loss.update_state(char_dist_loss)
        m.word_dist_loss.update_state(word_dist_loss)

        char_dist_loss = tf.nn.compute_average_loss(char_dist_loss, global_batch_size=self.global_batch_size)
        word_dist_loss = tf.nn.compute_average_loss(word_dist_loss, global_batch_size=self.global_batch_size)
        dist_loss = char_dist_loss + word_dist_loss


        # obj CE loss
        char_obj_loss = self.obj_loss(y_true=true_char_obj_whole, y_pred=pred_char_obj_whole)
        word_obj_loss = self.obj_loss(y_true=true_word_obj_whole, y_pred=pred_word_obj_whole)
        m.char_obj_loss.update_state(char_obj_loss)
        m.word_obj_loss.update_state(word_obj_loss)

        # for metric, only check true object matches
        m.char_obj_accuracy.update_state(true_char_obj, pred_char_obj)
        m.word_obj_accuracy.update_state(true_word_obj, pred_word_obj)

        char_obj_loss = tf.nn.compute_average_loss(char_obj_loss, global_batch_size=self.global_batch_size)
        word_obj_loss = tf.nn.compute_average_loss(word_obj_loss, global_batch_size=self.global_batch_size)
        obj_loss = char_obj_loss + word_obj_loss

        # char CE loss
        letters_ce_loss = self.letters_loss(y_true=true_char_letters, y_pred=pred_char_letters)
        m.letters_ce_loss.update_state(letters_ce_loss)
        m.letters_accuracy.update_state(true_char_letters, pred_char_letters)

        letters_ce_loss = tf.nn.compute_average_loss(letters_ce_loss, global_batch_size=self.global_batch_size)

        total_loss = dist_loss*0.1 + obj_loss + letters_ce_loss
        m.total_loss.update_state(total_loss)

        return total_loss
