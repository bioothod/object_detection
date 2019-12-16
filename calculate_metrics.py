import argparse
import json
import logging

import numpy as np

import coco

logger = logging.getLogger('detection')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

parser = argparse.ArgumentParser()
parser.add_argument('--true_annotations', type=str, help='Path to true annotations json in MS COCO format')
parser.add_argument('--pred_annotations', type=str, help='Path to inference results in json format')
FLAGS = parser.parse_args()

def calc_ious(pred_boxes, true_box):
    tx0 = true_box[0]
    ty0 = true_box[1]
    tx1 = true_box[2]
    ty1 = true_box[3]
    tarea = (tx1 - tx0) * (ty1 - ty0)

    px0 = pred_boxes[:, 0]
    py0 = pred_boxes[:, 1]
    px1 = pred_boxes[:, 2]
    py1 = pred_boxes[:, 3]
    parea = (px1 - px0) * (py1 - py0)

    xx0 = np.maximum(tx0, px0)
    yy0 = np.maximum(ty0, py0)
    xx1 = np.minimum(tx1, px1)
    yy1 = np.minimum(ty1, py1)

    w = np.maximum(0, xx1 - xx0 + 1)
    h = np.maximum(0, yy1 - yy0 + 1)

    inter = w * h
    ovr = inter / (tarea + parea - inter)
    return ovr

def convert_true_bbox_to_xyxy(true_bb):
    x0, y0, w, h = true_bb
    x1 = x0 + w
    y1 = y0 + h
    bb = [x0, y0, x1, y1]

    return np.array(bb)

class ClassMetric:
    def __init__(self, cid, cname, cat_ids):
        self.cname = cname
        self.cid = cid

        self.dexterity = np.zeros((100, len(cat_ids)))

    def feed_dexterity(self, score, cid):
        self.dexterity[int(score * 100), cid] += 1

class Metric:
    def __init__(self, prediction_json, cat_ids, cat_names):
        self.iou_threshold = 0.35
        self.obj_threshold = 0.3

        self.none_id = len(cat_ids)
        self.none_name = 'none'
        cat_ids.append(self.none_id)
        cat_names.append(self.none_name)

        self.cat_ids = cat_ids
        self.cat_names = cat_names

        # true/pred category id tuples
        self.cm_data = []
        self.cm = np.zeros(shape=[len(self.cat_ids), len(self.cat_ids)])

        self.cmetrics = {}
        for cid, cname in zip(cat_ids, cat_names):
            self.cmetrics[cid] = ClassMetric(cid, cname, self.cat_ids)

        self.fn2ann = self.open_predictions(prediction_json)

    def open_predictions(self, input_fn):
        fn2ann = {}
        with open(input_fn, 'r') as fin:
            results = json.load(fin)

        for js in results:
            fn = js['filename']
            anns = js['annotations']

            fn2ann[fn] = anns

        return fn2ann

    def feed_true_values(self, filename, true_anns):
        pred_anns = self.fn2ann[filename]
        num_pred_anns = len(pred_anns)

        pred_bboxes = []
        pred_objs = []
        pred_scores = []
        pred_cat_ids = []

        for pred_ann in pred_anns:
            # xyxy format
            pred_bb = pred_ann['bbox']
            pred_obj = pred_ann['objectness']
            pred_score = pred_ann['class_score']
            pred_cat_id = pred_ann['category_id']

            pred_bboxes.append(pred_bb)
            pred_objs.append(pred_obj)
            pred_scores.append(pred_score)
            pred_cat_ids.append(pred_cat_id)

        pred_bboxes = np.array(pred_bboxes)
        pred_objs = np.array(pred_objs)
        pred_scores = np.array(pred_scores)
        pred_cat_ids = np.array(pred_cat_ids)

        def good_name():
            return False
            return '1587110864_1094138' in filename

        if good_name():
            print('{}: true anns: {}'.format(filename, true_anns))
            print('{}: pred anns: {}'.format(filename, pred_anns))

        for true_ann in true_anns:
            # xywh format
            true_bb = true_ann['bbox']
            true_cat_id = true_ann['category_id']
            true_bb = convert_true_bbox_to_xyxy(true_bb)

            if len(pred_bboxes) == 0:
                self.cm_data.append((true_cat_id, self.none_id))
                self.cm[true_cat_id, self.none_id] += 1
                continue

            ious = calc_ious(pred_bboxes, true_bb)

            good_intersections_idx = np.where(ious > self.iou_threshold)[0]
            for good_idx in good_intersections_idx:
                pred_ann = pred_anns[good_idx]

                pred_obj = pred_ann['objectness']
                pred_cat_id = pred_ann['category_id']
                pred_score = pred_ann['class_score']

                if pred_obj < self.obj_threshold:
                    continue

                self.cmetrics[true_cat_id].feed_dexterity(pred_score, pred_cat_id)

            if good_name():
                print('{}: 1 true anns: {}, pred anns: {} -> {}'.format(filename, len(true_anns), num_pred_anns, len(pred_anns)))

            good_idx = np.where((ious >= self.iou_threshold) & (pred_objs >= self.obj_threshold))[0]
            if len(good_idx) == 0:
                if good_name():
                    print('{}: 2 true anns: {}, pred anns: {} -> {}: ious: {}, pred_objs: {}'.format(filename, len(true_anns), num_pred_anns, len(pred_anns), ious, pred_objs))
                self.cm_data.append((true_cat_id, self.none_id))
                self.cm[true_cat_id, self.none_id] += 1
                continue

            good_scores = pred_scores[good_idx]
            good_objs = pred_objs[good_idx]
            good_ious = ious[good_idx]
            good_cat_ids = pred_cat_ids[good_idx]

            scores = good_ious * good_objs * good_scores
            best_score_idx = np.argmax(scores)
            best_score_idx = good_idx[best_score_idx]

            best_iou = ious[best_score_idx]
            best_obj = pred_objs[best_score_idx]

            pred_ann = pred_anns.pop(best_score_idx)
            pred_bboxes = np.delete(pred_bboxes, best_score_idx, axis=0)
            pred_objs = np.delete(pred_objs, best_score_idx, axis=0)
            pred_scores = np.delete(pred_scores, best_score_idx, axis=0)
            pred_cat_ids = np.delete(pred_cat_ids, best_score_idx, axis=0)

            pred_obj = pred_ann['objectness']
            if pred_obj < self.obj_threshold:
                raise ValueError('predicted objectness is less than threshold')

            pred_cat_id = pred_ann['category_id']
            pred_score = pred_ann['class_score']
            pred_obj = pred_ann['objectness']

            if good_name():
                print('{}: true label: {}, iou: {}, label: {}, objs: {}, score: {}'.format(filename, true_cat_id, best_iou, pred_cat_id, pred_obj, pred_score))


            self.cm_data.append((true_cat_id, pred_cat_id))
            self.cm[true_cat_id, pred_cat_id] += 1

        for pred_ann in pred_anns:
            pred_cat_id = pred_ann['category_id']

            self.cm_data.append((self.none_id, pred_cat_id))
            self.cm[self.none_id, pred_cat_id] += 1

    def confusion_matrix(self):
        return self.cm.astype(int)

    def per_entry_stats(self):
        orig_cm = self.confusion_matrix()
        cm = orig_cm[:-1, :-1]

        sum_pred = np.sum(cm, axis=0)
        sum_true = np.sum(cm, axis=1)

        for cid, cname in zip(self.cat_ids[:-1], self.cat_names[:-1]):
            tp = cm[cid][cid]

            fp = sum_pred[cid] - tp
            fn = sum_true[cid] - tp

            rc = tp / (tp + fn + 1e-10) * 100
            pr = tp / (tp + fp + 1e-10) * 100
            f1 = 2 * pr * rc / (pr + rc + 1e-10)

            undetected = orig_cm[cid][self.none_id] / (np.sum(orig_cm[cid, :]) + 1e-10) * 100
            unlabeled = orig_cm[self.none_id][cid] / (np.sum(orig_cm[:, cid]) + 1e-10) * 100

            print('{: >20s}/{:2d}: f1: {:.1f}, precision: {:.1f}, recall: {:.1f}, tp: {:4d}, fp: {:4d}, fn: {:4d}, undetected: {:.1f}, unlabeled: {:.1f}'.format(
                cname, cid, f1, pr, rc, tp, fp, fn,
                undetected, unlabeled))


def main():
    np.set_printoptions(formatter={'float': '{:0.4f}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    true_base = coco.COCO(FLAGS.true_annotations, '', logger)
    true_image_tuples = true_base.get_images()

    num_images = len(true_image_tuples)
    cat_names = []
    cat_ids = []

    from scan_tags import categories
    for cat_name, cat_obj in categories.items():
        cat_names.append(cat_name)
        cat_id = cat_obj['id']
        cat_ids.append(cat_id)

    metric = Metric(FLAGS.pred_annotations, cat_ids, cat_names)

    for i in range(num_images):
        filename, image_id, anns = true_image_tuples[i]

        metric.feed_true_values(filename, anns)
    
    cm = metric.confusion_matrix()
    print(cm)
    metric.per_entry_stats()


if __name__ == '__main__':
    main()
