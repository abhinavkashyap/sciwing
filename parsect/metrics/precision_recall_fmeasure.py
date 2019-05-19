import torch
from typing import Dict
from wasabi import Printer
from wasabi import table
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from parsect.utils.common import merge_dictionaries_with_sum
import numpy as np
from collections import Counter


class PrecisionRecallFMeasure:
    def __init__(self):
        self.msg_printer = Printer()

        # setup counters to calculate true positives, false positives,
        # false negatives and true negatives
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def get_overall_accuracy(self, predicted_probs: torch.FloatTensor,
                             labels: torch.LongTensor) -> Dict[str, Dict[int, float]]:
        """
        NOTE: Please use this only when you want the precision recall and f measure
        of the overall dataset.
        2. If you want to calculate metrics per batch, there is another method
        in this file which calculates true_positives, false_positives,
        true_negatives and false negatives per class
        :param predicted_probs: type: torch.FloatTensor
                                shape  N * C
                                N - Batch size
                                C - Number of classes
        Predicted Probabilities for different classes
        :param labels: type: torch.LongTensor
                      shape: N,
                      N - batch size
        The correct class for different instances of the batch
        :return:
        """
        assert predicted_probs.ndimension() == 2, self.msg_printer.fail("The predicted probs should "
                                                                        "have 2 dimensions. The probs "
                                                                        "that you passed have shape "
                                                                        "{0}".format(predicted_probs.size()))

        assert labels.ndimension() == 1, self.msg_printer.fail("The labels should have 1 dimension."
                                                               "The labels that you passed have shape "
                                                               "{0}".format(labels.size()))

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = predicted_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.numpy().ravel()

        # convert labels to 1 dimension
        labels_numpy = labels.numpy()

        # average: None gives per class precision, recall, fscore and support
        precision, recall, fscore, support = precision_recall_fscore_support(labels_numpy,
                                                                             top_indices_numpy,
                                                                             average=None)

        metrics = {}

        classes = unique_labels(labels_numpy, top_indices_numpy)
        classes = classes.tolist()

        precision_list = precision.tolist()
        recall_list = recall.tolist()
        fscore_list = fscore.tolist()

        label_precision_map = dict(zip(classes, precision_list))
        label_recall_map = dict(zip(classes, recall_list))
        label_fscore_map = dict(zip(classes, fscore_list))

        metrics['precision'] = label_precision_map
        metrics['recall'] = label_recall_map
        metrics['fscore'] = label_fscore_map

        return metrics

    def print_confusion_metrics(self, predicted_probs: torch.FloatTensor,
                                labels: torch.LongTensor) -> None:

        assert predicted_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(predicted_probs.size()))

        assert labels.ndimension() == 1, self.msg_printer.fail("The labels should have 1 dimension."
                                                               "The labels that you passed have shape "
                                                               "{0}".format(labels.size()))

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = predicted_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.numpy().ravel()

        # convert labels to 1 dimension
        true_labels_numpy = labels.numpy()

        confusion_mtrx = confusion_matrix(true_labels_numpy, top_indices_numpy)

        classes = unique_labels(true_labels_numpy, top_indices_numpy)
        classes = classes.tolist()

        # insert th
        confusion_mtrx = np.insert(confusion_mtrx, 0, classes, axis=1)

        assert len(classes) == confusion_mtrx.shape[1] - 1

        header = ['{0}'.format(class_) for class_ in classes]
        header.insert(0, 'pred (cols) / true (rows)')

        self.msg_printer.table(data=confusion_mtrx,
                               header=header,
                               divider=True)

    def calc_metric(self, predicted_probs: torch.FloatTensor,
                    labels: torch.LongTensor) -> None :

        assert predicted_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(predicted_probs.size()))

        assert labels.ndimension() == 1, self.msg_printer.fail("The labels should have 1 dimension."
                                                               "The labels that you passed have shape "
                                                               "{0}".format(labels.size()))

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = predicted_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.numpy().ravel()

        # convert labels to 1 dimension
        labels_numpy = labels.numpy()

        confusion_mtrx = confusion_matrix(labels_numpy, top_indices_numpy)

        classes = unique_labels(labels_numpy, top_indices_numpy)
        classes = classes.tolist()

        # calculate tps
        tps = np.diag(confusion_mtrx)

        # calculate fps
        fps = np.sum(confusion_mtrx, axis=0) - tps

        # calculate fns
        fns = np.sum(confusion_mtrx, axis=1) - tps

        tps = tps.tolist()
        fps = fps.tolist()
        fns = fns.tolist()

        class_tps_mapping = dict(zip(classes, tps))
        class_fps_mapping = dict(zip(classes, fps))
        class_fns_mapping = dict(zip(classes, fns))

        self.tp_counter = merge_dictionaries_with_sum(self.tp_counter, class_tps_mapping)
        self.fp_counter = merge_dictionaries_with_sum(self.fp_counter, class_fps_mapping)
        self.fn_counter = merge_dictionaries_with_sum(self.fn_counter, class_fns_mapping)

    def get_metric(self) -> Dict[str, Dict[str, float]]:
        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}

        for key in self.tp_counter.keys():
            tp = self.tp_counter[key]
            fp = self.fp_counter[key]
            fn = self.fn_counter[key]
            if tp == 0 and fp == 0:
                precision = 0
                self.msg_printer.warn("both tp and fp are 0 .. setting precision to 0")
            else:
                precision = tp / (tp + fp)
            if tp == 0 and fn == 0:
                recall = 0
                self.msg_printer.warn("both tp and fn are 0 .. setting recall to 0")
            else:
                recall = tp / (tp + fn)

            if precision == 0 and recall == 0:
                fscore = 0
                self.msg_printer.warn("both precision and recall are 0 .. setting fscore to 0")
            else:
                fscore = (2 * precision * recall) / (precision + recall)

            precision_dict[key] = precision
            recall_dict[key] = recall
            fscore_dict[key] = fscore

        return {
            'precision': precision_dict,
            'recall': recall_dict,
            'fscore': fscore_dict
        }

    def reset(self) -> None:
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def report_metrics(self,
                      report_type="wasabi"):

        accuracy_metrics = self.get_metric()
        precision = accuracy_metrics['precision']
        recall = accuracy_metrics['recall']
        fscore = accuracy_metrics['fscore']

        if report_type == 'wasabi':
            classes = precision.keys()
            classes = sorted(classes)
            header_row = [' ', 'Precision', 'Recall', 'F_measure']
            rows = []
            for class_num in classes:
                p = precision[class_num]
                r = recall[class_num]
                f = fscore[class_num]
                rows.append(('class_{0}'.format(class_num), p, r, f))

            return table(rows, header=header_row, divider=True)


if __name__ == '__main__':
    predicted_probs = torch.FloatTensor([[0.8, 0.1, 0.2],
                                         [0.2, 0.5, 0.3]])
    labels = torch.LongTensor([0, 2])

    accuracy = PrecisionRecallFMeasure()

    accuracy.calc_metric(predicted_probs, labels)
    metrics_ = accuracy.get_metric()
    precision_ = metrics_['precision']
    recall_ = metrics_['recall']
    fscore_ = metrics_['fscore']
    print('precision', precision_)
    print('recall', recall_)
    print('fmeasure', fscore_)
