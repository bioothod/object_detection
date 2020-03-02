import logging
logger = logging.getLogger('detection')

import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, label_smoothing=0, sigmoid_ce=False, reduction=tf.keras.losses.Reduction.NONE, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = 0.25
        self.gamma = 2
        self.reduction = reduction
        self.epsilon = 1e-9
        self.from_logits = from_logits
        self.sigmoid_ce = sigmoid_ce

    def call(self, y_true, y_pred, weights=None):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            if self.sigmoid_ce:
                y_pred = tf.nn.sigmoid(y_pred)
            else:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.clip_by_value(y_pred, self.epsilon, 1. - self.epsilon)

        ce = tf.multiply(y_true, -tf.math.log(y_pred))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), self.gamma))
        fl = tf.multiply(self.alpha, tf.multiply(weight, ce))
        if self.reduction == tf.keras.losses.Reduction.NONE:
            fl = tf.reduce_sum(fl, -1)
        else:
            fl = tf.reduce_sum(fl)

        return fl

class TextMetric:
    def __init__(self, max_sequence_len, dictionary_size, label_smoothing=0, from_logits=False, **kwargs):
        self.dictionary_size = dictionary_size
        self.max_sequence_len = max_sequence_len

        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)
        #self.ce = FocalLoss(from_logits=from_logits, label_smoothing=label_smoothing, sigmoid_ce=False, reduction=tf.keras.losses.Reduction.NONE, name='text_focal_loss')

        self.word_loss = tf.keras.metrics.Mean()
        self.full_loss = tf.keras.metrics.Mean()

        self.word3_acc = tf.keras.metrics.CategoricalAccuracy()
        self.word_acc = tf.keras.metrics.CategoricalAccuracy()
        self.full_acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, true_texts, true_lengths, logits):
        dtype = tf.float32
        logits = tf.cast(logits, dtype)

        true_texts_oh = tf.one_hot(true_texts, self.dictionary_size, dtype=dtype)

        batch_size = tf.shape(logits)[0]
        weights = tf.range(self.max_sequence_len, dtype=true_lengths.dtype)
        weights = tf.expand_dims(weights, 0)
        weights = tf.tile(weights, [batch_size, 1])

        true_lengths = tf.expand_dims(true_lengths, 1)
        true_lengths = tf.tile(true_lengths, [1, self.max_sequence_len])
        #logger.info('true_texts: {}, true_texts_oh: {}, logits: {}, weights: {}'.format(true_texts.shape, true_texts_oh.shape, logits.shape, weights.shape))
        weights_word = tf.where(weights < true_lengths, tf.ones_like(weights, dtype=dtype), tf.zeros_like(weights, dtype=dtype))
        weights_3 = tf.where(weights < 3, tf.ones_like(weights, dtype=dtype), tf.zeros_like(weights, dtype=dtype))

        word_loss = self.ce(y_true=true_texts_oh, y_pred=logits, sample_weight=weights_word)
        full_loss = self.ce(y_true=true_texts_oh, y_pred=logits)

        self.word_acc.update_state(true_texts_oh, logits, sample_weight=weights_word)
        self.word3_acc.update_state(true_texts_oh, logits, sample_weight=weights_3)
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
            return '{:.4f}/{:.4f}/{:.4f}'.format(
                self.word3_acc.result(),
                self.word_acc.result(),
                self.full_acc.result())

        return ''

    def reset_states(self):
        self.word_loss.reset_states()
        self.full_loss.reset_states()

        self.word3_acc.reset_states()
        self.word_acc.reset_states()
        self.full_acc.reset_states()


class Metric:
    def __init__(self, max_sequence_len, dictionary_size, from_logits=False, training=True, label_smoothing=0.1, **kwargs):
        self.total_loss = tf.keras.metrics.Mean()

        self.training = training

        self.word_dist_loss = tf.keras.metrics.Mean()

        self.word_obj_loss = tf.keras.metrics.Mean()
        self.word_obj_whole_loss = tf.keras.metrics.Mean()
        self.word_obj_accuracy02 = tf.keras.metrics.BinaryAccuracy(threshold=0.2)
        self.word_obj_accuracy05 = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.word_obj_whole_accuracy05 = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

        self.text_metric = TextMetric(max_sequence_len, dictionary_size, label_smoothing=label_smoothing, from_logits=from_logits)
        self.text_metric_ar = TextMetric(max_sequence_len, dictionary_size, label_smoothing=label_smoothing, from_logits=from_logits)

    def reset_states(self):
        self.total_loss.reset_states()

        self.word_dist_loss.reset_states()

        self.word_obj_loss.reset_states()
        self.word_obj_whole_loss.reset_states()
        self.word_obj_accuracy02.reset_states()
        self.word_obj_accuracy05.reset_states()
        self.word_obj_whole_accuracy05.reset_states()

        self.text_metric.reset_states()
        self.text_metric_ar.reset_states()

    def str_result(self):
        if self.training:
            return 'total_loss: {:.3e}, dist: {:.3f}, acc: {}, AR acc: {}, word_obj_acc: {:.3f}/{:.3f}/{:.4f}'.format(
                    self.total_loss.result(),
                    self.word_dist_loss.result(),

                    self.text_metric.str_result(want_acc=True),

                    self.text_metric_ar.str_result(want_acc=True),

                    self.word_obj_accuracy02.result(),
                    self.word_obj_accuracy05.result(),
                    self.word_obj_whole_accuracy05.result(),
                    )
        else:
            return 'total_loss: {:.3e}, dist: {:.3f}, acc: {}, word_obj_acc: {:.3f}/{:.4f}/{:.4f}'.format(
                    self.total_loss.result(),
                    self.word_dist_loss.result(),

                    self.text_metric_ar.str_result(want_acc=True),

                    self.word_obj_accuracy02.result(),
                    self.word_obj_accuracy05.result(),
                    self.word_obj_whole_accuracy05.result(),
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
        #self.obj_loss = FocalLoss(label_smoothing=label_smoothing, from_logits=True, sigmoid_ce=True, reduction=tf.keras.losses.Reduction.NONE, name='obj_focal_loss')
        self.obj_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing, from_logits=True, reduction=tf.keras.losses.Reduction.NONE, name='obj_focal_loss')

        self.train_metric = Metric(max_sequence_len, dictionary_size, from_logits=False, label_smoothing=label_smoothing, name='train_metric', training=True)
        self.eval_metric = Metric(max_sequence_len, dictionary_size, from_logits=False, label_smoothing=label_smoothing, name='eval_metric', training=False)

    def str_result(self, training):
        m = self.train_metric
        if not training:
            m = self.eval_metric

        return m.str_result()

    def evaluation_result(self):
        m = self.eval_metric
        obj_acc = m.word_obj_accuracy02.result()
        word_acc, full_acc = m.text_metric_ar.result(want_acc=True)

        return word_acc

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def object_detection_loss(self, word_object_mask, true_word_poly, y_pred, training):
        # predicted tensors
        pred_word_obj = y_pred[..., 0]
        pred_word_poly = y_pred[..., 1 : 9]

        pred_word_obj = tf.cast(pred_word_obj, tf.float32)
        pred_word_poly = tf.cast(pred_word_poly, tf.float32)

        true_word_obj = tf.cast(word_object_mask, tf.float32)

        true_word_obj_whole = true_word_obj
        pred_word_obj_whole = pred_word_obj

        pred_word_obj = tf.boolean_mask(pred_word_obj, word_object_mask)
        pred_word_poly = tf.boolean_mask(pred_word_poly, word_object_mask)

        true_word_obj = tf.boolean_mask(true_word_obj, word_object_mask)
        true_word_poly = tf.boolean_mask(true_word_poly, word_object_mask)


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
        word_obj_loss = word_obj_loss / self.global_batch_size

        obj_loss = word_obj_loss + word_obj_whole_loss

        # for accuracy metric, only check true object matches
        m.word_obj_accuracy02.update_state(true_word_obj, pred_word_obj)
        m.word_obj_accuracy05.update_state(true_word_obj, pred_word_obj)
        m.word_obj_whole_accuracy05.update_state(true_word_obj_whole, pred_word_obj_whole)

        def nans(p):
            pred_obj_nan = tf.math.is_nan(p)
            pred_obj_nan = tf.cast(pred_obj_nan, tf.int32)
            pred_obj_nans = tf.reduce_sum(pred_obj_nan)
            return pred_obj_nans

        pred_obj_nans = nans(pred_word_obj)
        pred_word_poly_nans = nans(pred_word_poly)
        pred_obj_whole_nans = nans(pred_word_obj_whole)
        if pred_obj_nans > 0:
            tf.print('pred_word_obj:', pred_word_obj)
            tf.print('pred_obj_nans:', pred_obj_nans, 'pred_obj_whole_nans:', pred_obj_whole_nans, 'pred_word_poly_nans:', pred_word_poly_nans)

        return dist_loss + obj_loss

    def text_recognition_loss(self, word_object_mask, true_words, true_lengths, y_pred_rnn, y_pred_rnn_ar, training):
        true_words = tf.boolean_mask(true_words, word_object_mask)
        true_lengths = tf.boolean_mask(true_lengths, word_object_mask)

        m = self.train_metric
        if not training:
            m = self.eval_metric

        # text CE loss
        if training:
            word_ce_loss, full_ce_loss = m.text_metric.update_state(true_words, true_lengths, y_pred_rnn)
            text_ce_loss = word_ce_loss + full_ce_loss*0.01
            text_ce_loss = tf.nn.compute_average_loss(text_ce_loss, global_batch_size=self.global_batch_size)
        else:
            text_ce_loss = 0

        word_ce_loss_ar, full_ce_loss_ar = m.text_metric_ar.update_state(true_words, true_lengths, y_pred_rnn_ar)
        text_ce_loss_ar = word_ce_loss_ar + full_ce_loss_ar*0.01
        text_ce_loss_ar = tf.nn.compute_average_loss(text_ce_loss_ar, global_batch_size=self.global_batch_size)

        total_loss = text_ce_loss + text_ce_loss_ar

        return total_loss
