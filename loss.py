import logging
logger = logging.getLogger('detection')

import tensorflow as tf

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

        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = tf.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = tf.where(y_true > zeros, zeros, sigmoid_p)

        p = tf.clip_by_value(sigmoid_p, 1e-10, 1)
        one_minus_p = tf.clip_by_value(1 - sigmoid_p, 1e-10, 1)

        per_entry_cross_ent = -self.alpha * (pos_p_sub ** self.gamma) * tf.math.log(p) - \
                (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.math.log(one_minus_p)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return per_entry_cross_ent

        raise "qwe"


class TextMetric():
    def __init__(self, max_sequence_len, dictionary_size, label_smoothing=0, from_logits=False, **kwargs):
        self.dictionary_size = dictionary_size
        self.max_sequence_len = max_sequence_len

        self.from_logits = from_logits

        #self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)
        self.ce = FocalLoss(from_logits=False, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)

        self.word_loss = tf.keras.metrics.Mean()
        self.full_loss = tf.keras.metrics.Mean()

        self.word_acc = tf.keras.metrics.CategoricalAccuracy()
        self.full_acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, true_texts, true_lengths, logits, cur_max_sequence_len):
        true_texts_oh = tf.one_hot(true_texts, self.dictionary_size)

        batch_size = tf.shape(logits)[0]
        weights = tf.range(self.max_sequence_len, dtype=true_lengths.dtype)
        weights = tf.expand_dims(weights, 0)
        weights = tf.tile(weights, [batch_size, 1])

        true_lengths = tf.expand_dims(true_lengths, 1)
        true_lengths = tf.tile(true_lengths, [1, self.max_sequence_len])
        #logger.info('true_texts: {}, true_texts_oh: {}, logits: {}, weights: {}'.format(true_texts.shape, true_texts_oh.shape, logits.shape, weights.shape))
        weights = tf.where(weights < true_lengths, tf.ones_like(weights), tf.zeros_like(weights))

        if self.from_logits:
            logits = tf.nn.softmax(logits, -1)

        word_loss = self.ce(y_true=true_texts_oh, y_pred=logits, sample_weight=weights)
        full_loss = self.ce(y_true=true_texts_oh, y_pred=logits)

        self.word_acc.update_state(true_texts_oh, logits, sample_weight=weights)
        self.full_acc.update_state(true_texts_oh, logits)

        self.word_loss.update_state(word_loss)
        self.full_loss.update_state(full_loss)

        return word_loss, full_loss

    def result(self, want_loss=False, want_acc=False):
        if want_loss:
            return self.word_loss.result(), self.full_loss.result()

        if want_acc:
            return self.word_acc.result(), self.full_acc.result()

        return None

    def str_result(self, want_loss=False, want_acc=False):
        if want_loss:
            return '{:.3f}/{:.3f}'.format(
                self.word_loss.result(),
                self.full_loss.result())

        if want_acc:
            return '{:.4f}/{:.4f}'.format(
                self.word_acc.result(),
                self.full_acc.result())

        return ''

    def reset_states(self):
        self.word_loss.reset_states()
        self.full_loss.reset_states()

        self.word_acc.reset_states()
        self.full_acc.reset_states()


class Metric:
    def __init__(self, max_sequence_len, dictionary_size, from_logits=False, training=True, **kwargs):
        self.total_loss = tf.keras.metrics.Mean()

        self.training = training

        self.word_dist_loss = tf.keras.metrics.Mean()
        self.word_obj_loss = tf.keras.metrics.Mean()
        self.word_obj_whole_loss = tf.keras.metrics.Mean()
        self.word_obj_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.2)

        self.text_metric = TextMetric(max_sequence_len, dictionary_size, label_smoothing=0.1, from_logits=from_logits)
        self.text_metric_ar = TextMetric(max_sequence_len, dictionary_size, label_smoothing=0.1, from_logits=from_logits)

    def reset_states(self):
        self.total_loss.reset_states()

        self.word_dist_loss.reset_states()

        self.word_obj_loss.reset_states()

        self.text_metric.reset_states()
        self.text_metric_ar.reset_states()

        self.word_obj_accuracy.reset_states()

    def str_result(self):
        if self.training:
            return 'total_loss: {:.3f}, dist: {:.3f}, text ce: {}, acc: {}, AR text ce: {}, acc: {}, word_obj_acc: {:.4f}'.format(
                    self.total_loss.result(),
                    self.word_dist_loss.result(),

                    self.text_metric.str_result(want_loss=True),
                    self.text_metric.str_result(want_acc=True),

                    self.text_metric_ar.str_result(want_loss=True),
                    self.text_metric_ar.str_result(want_acc=True),

                    self.word_obj_accuracy.result(),
                    )
        else:
            return 'total_loss: {:.3f}, dist: {:.3f}, AR text ce: {}, acc: {}, word_obj_acc: {:.4f}'.format(
                    self.total_loss.result(),
                    self.word_dist_loss.result(),

                    self.text_metric_ar.str_result(want_loss=True),
                    self.text_metric_ar.str_result(want_acc=True),

                    self.word_obj_accuracy.result(),
                    )


class LossMetricAggregator:
    def __init__(self,
                 max_sequence_len, dictionary_size,
                 global_batch_size,
                 **kwargs):

        self.global_batch_size = global_batch_size
        self.max_sequence_len = max_sequence_len
        self.dictionary_size = dictionary_size

        label_smoothing = 0.1

        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.obj_loss = FocalLoss(label_smoothing=label_smoothing, from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.train_metric = Metric(max_sequence_len, dictionary_size, from_logits=True, label_smoothing=label_smoothing, name='train_metric', training=True)
        self.eval_metric = Metric(max_sequence_len, dictionary_size, from_logits=True, label_smoothing=label_smoothing, name='eval_metric', training=False)

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def evaluation_result(self):
        m = self.eval_metric
        obj_acc = m.word_obj_accuracy.result()
        dist = tf.math.exp(-m.word_dist_loss.result())
        word_acc, full_acc = m.text_metric_ar.result(want_acc=True)

        return obj_acc + dist + word_acc

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

    def loss(self, y_true, y_pred, y_pred_rnn, y_pred_rnn_ar, current_max_sequence_len, training):
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

        pred_obj_nan = tf.math.is_nan(pred_word_obj)
        pred_obj_nan = tf.cast(pred_obj_nan, tf.int32)
        pred_obj_nans = tf.reduce_sum(pred_obj_nan)
        if pred_obj_nans > 0:
            tf.print('pred_word_obj:', pred_word_obj)
            tf.print('pred_obj_nans:', pred_obj_nans)

        # text CE loss
        if training:
            word_ce_loss, full_ce_loss = m.text_metric.update_state(true_words, true_lengths, y_pred_rnn, current_max_sequence_len)
            text_ce_loss = word_ce_loss*10 + full_ce_loss
            text_ce_loss = tf.nn.compute_average_loss(text_ce_loss, global_batch_size=self.global_batch_size)
        else:
            text_ce_loss = 0

        word_ce_loss_ar, full_ce_loss_ar = m.text_metric_ar.update_state(true_words, true_lengths, y_pred_rnn_ar, current_max_sequence_len)
        text_ce_loss_ar = word_ce_loss_ar*10 + full_ce_loss_ar
        text_ce_loss_ar = tf.nn.compute_average_loss(text_ce_loss_ar, global_batch_size=self.global_batch_size)

        total_loss = dist_loss + 10*obj_loss + text_ce_loss + text_ce_loss_ar
        m.total_loss.update_state(total_loss)

        return total_loss
