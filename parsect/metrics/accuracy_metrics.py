import torch
from typing import Dict
from wasabi import Printer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


class Accuracy:
    def __init__(self):
        self.msg_printer = Printer()

    def get_accuracy(self, predicted_probs: torch.FloatTensor,
                     labels: torch.LongTensor) -> Dict[str, Dict[int, float]]:
        """

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

        # get the set of labels in the batch of predictions
        labels_set = set(labels_numpy.tolist())
        top_indices_set = set(top_indices_numpy.tolist())

        print("labels set", labels_set)
        print('top_indices set', top_indices_set)

        classes = labels_set.union(top_indices_set)
        classes = sorted(list(classes))

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
        labels_numpy = labels.numpy()

        confusion_mtrx = confusion_matrix(labels_numpy, top_indices_numpy)

        # get the set of labels in the batch of predictions
        labels_set = set(labels_numpy.tolist())
        top_indices_set = set(top_indices_numpy.tolist())
        classes = labels_set.union(top_indices_set)
        classes = sorted(list(classes))

        assert len(classes) == len(confusion_mtrx.tolist())

        header = classes

        self.msg_printer.table(data=confusion_mtrx,
                               header=header,
                               divider=True)

