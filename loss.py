import logging
logger = logging.getLogger('detection')

import tensorflow as tf
from tensorflow.python.ops import array_ops

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2
        self.reduction = reduction
        self.from_logits = from_logits

    def call(self, y_true, y_pred, weights=None):
        y_true = tf.clip_by_value(y_true, 0, 1)

        if self.from_logits:
            sigmoid_p = tf.nn.sigmoid(y_pred)
        else:
            sigmoid_p = y_pred

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)

        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1)) - \
            (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1))

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_cross_ent

        raise "qwe"


class TextMetric():
    def __init__(self, max_sequence_len, dictionary_size, name=None, label_smoothing=0, from_logits=False, **kwargs):
        self.dictionary_size = dictionary_size
        self.max_sequence_len = max_sequence_len

        #self.ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)
        self.ce = FocalLoss(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)

        self.loss = tf.keras.metrics.Mean()
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, true_texts, logits, cur_max_sequence_len):
        true_texts_oh = tf.one_hot(true_texts, self.dictionary_size)

        if True:
            batch_size = tf.shape(logits)[0]
            weights = tf.range(self.max_sequence_len, dtype=cur_max_sequence_len.dtype)
            weights = tf.expand_dims(weights, 0)
            weights = tf.tile(weights, [batch_size, 1])
            #logger.info('true_texts: {}, true_texts_oh: {}, logits: {}, weights: {}'.format(true_texts.shape, true_texts_oh.shape, logits.shape, weights.shape))
            weights = tf.where(weights < cur_max_sequence_len, tf.ones_like(weights), tf.zeros_like(weights))

            ce_loss = self.ce(y_true=true_texts_oh, y_pred=logits, sample_weight=weights)
            self.acc.update_state(true_texts_oh, logits, sample_weight=weights)
        else:
            ce_loss = self.ce(y_true=true_texts_oh, y_pred=logits)
            self.acc.update_state(true_texts, logits)

        self.loss.update_state(ce_loss)

        return ce_loss

    def result(self, want_loss=False, want_acc=False):
        if want_loss:
            return self.loss.result()
        if want_acc:
            return self.acc.result()

        return None

    def str_result(self, want_cm=False):
        ms = 'loss: {:.3e}, acc: {:.4f}'.format(self.loss.result(), self.acc.result())

        return ms

    def reset_states(self):
        self.loss.reset_states()
        self.acc.reset_states()


class Metric:
    def __init__(self, max_sequence_len, dictionary_size, **kwargs):
        self.total_loss = tf.keras.metrics.Mean()

        self.word_dist_loss = tf.keras.metrics.Mean()
        self.word_obj_loss = tf.keras.metrics.Mean()
        self.word_obj_whole_loss = tf.keras.metrics.Mean()
        self.word_obj_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.word_obj_whole_accuracy = tf.keras.metrics.BinaryAccuracy()

        self.text_metric = TextMetric(max_sequence_len, dictionary_size, label_smoothing=0.1)

    def reset_states(self):
        self.total_loss.reset_states()

        self.word_dist_loss.reset_states()

        self.word_obj_loss.reset_states()

        self.text_metric.reset_states()

        self.word_obj_accuracy.reset_states()
        self.word_obj_whole_accuracy.reset_states()

    def str_result(self):
        return 'total_loss: {:.3e}, dist: {:.3e}, obj: {:.3e}/{:.3e}, text_ce: {:.3e}, text_acc: {:.4f}, word_obj_acc: {:.5f}/{:.5f}'.format(
                self.total_loss.result(),
                self.word_dist_loss.result(),

                self.word_obj_loss.result(),
                self.word_obj_whole_loss.result(),

                self.text_metric.result(want_loss=True),
                self.text_metric.result(want_acc=True),

                self.word_obj_accuracy.result(),
                self.word_obj_whole_accuracy.result(),
                )


class LossMetricAggregator:
    def __init__(self,
                 max_sequence_len, dictionary_size,
                 global_batch_size,
                 **kwargs):

        #super(YOLOLoss, self).__init__(**kwargs)

        self.global_batch_size = global_batch_size
        self.max_sequence_len = max_sequence_len
        self.dictionary_size = dictionary_size

        label_smoothing = 0.1

        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.obj_loss = FocalLoss(label_smoothing=label_smoothing, from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.train_metric = Metric(max_sequence_len, dictionary_size, from_logits=True, label_smoothing=label_smoothing, name='train_metric')
        self.eval_metric = Metric(max_sequence_len, dictionary_size, from_logits=True, label_smoothing=label_smoothing, name='eval_metric')

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def evaluation_result(self):
        m = self.eval_metric
        obj_acc = m.word_obj_accuracy.result()
        dist = tf.math.exp(-m.word_dist_loss.result())
        text_acc = m.text_metric.result(want_acc=True)

        return obj_acc + dist + text_acc

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

    def loss(self, y_true, y_pred, y_pred_rnn, current_max_sequence_len, training):
        # predicted tensors
        pred_word_obj = y_pred[..., 0]
        pred_word_poly = y_pred[..., 1 : 9]

        # true tensors
        true_word_obj = y_true[..., 0]
        true_word_poly = y_true[..., 1 : 9]
        true_words = y_true[..., 9 : 9 + self.max_sequence_len]
        true_lengths = y_true[..., 9 + self.max_sequence_len]

        true_words = tf.cast(true_words, tf.int64)
        true_lengths = tf.cast(true_lengths, tf.int64)

        true_word_obj_whole = true_word_obj
        pred_word_obj_whole = pred_word_obj

        word_object_mask = tf.cast(true_word_obj, 'bool')


        pred_word_obj = tf.boolean_mask(pred_word_obj, word_object_mask)
        pred_word_poly = tf.boolean_mask(pred_word_poly, word_object_mask)


        true_word_obj = tf.boolean_mask(true_word_obj, word_object_mask)
        true_word_poly = tf.boolean_mask(true_word_poly, word_object_mask)
        true_words = tf.boolean_mask(true_words, word_object_mask)
        true_lengths = tf.boolean_mask(true_lengths, word_object_mask)

        m = self.train_metric
        if not training:
            m = self.eval_metric


        # losses

        # distance loss
        word_dist_loss = self.mae(true_word_poly, pred_word_poly)
        m.word_dist_loss.update_state(word_dist_loss)
        word_dist_loss = tf.nn.compute_average_loss(word_dist_loss, global_batch_size=self.global_batch_size)
        dist_loss = word_dist_loss


        # obj CE loss
        word_obj_whole_loss = self.obj_loss(y_true=true_word_obj_whole, y_pred=pred_word_obj_whole)
        m.word_obj_whole_loss.update_state(word_obj_whole_loss)
        word_obj_whole_loss = tf.nn.compute_average_loss(word_obj_whole_loss, global_batch_size=self.global_batch_size)

        word_obj_loss = self.obj_loss(y_true=true_word_obj, y_pred=pred_word_obj)
        m.word_obj_loss.update_state(word_obj_loss)
        word_obj_loss = tf.nn.compute_average_loss(word_obj_loss, global_batch_size=self.global_batch_size)
        obj_loss = word_obj_loss + word_obj_whole_loss

        # for accuracy metric, only check true object matches
        m.word_obj_accuracy.update_state(true_word_obj, pred_word_obj)
        m.word_obj_whole_accuracy.update_state(true_word_obj_whole, pred_word_obj_whole)


        # text CE loss
        pred_words = y_pred_rnn
        text_ce_loss = m.text_metric.update_state(true_words, pred_words, current_max_sequence_len)
        text_ce_loss = tf.nn.compute_average_loss(text_ce_loss, global_batch_size=self.global_batch_size)

        total_loss = dist_loss + 10*obj_loss + text_ce_loss
        m.total_loss.update_state(total_loss)

        return total_loss
