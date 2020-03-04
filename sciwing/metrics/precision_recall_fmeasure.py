import torch
from typing import Dict, Union, Any, Optional, List
from wasabi import Printer
from sciwing.utils.common import merge_dictionaries_with_sum
from sciwing.data.line import Line
from sciwing.data.label import Label
import numpy as np
import pandas as pd
from sciwing.metrics.BaseMetric import BaseMetric
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.metrics.classification_metrics_utils import ClassificationMetricsUtils
from sciwing.utils.class_nursery import ClassNursery


class PrecisionRecallFMeasure(BaseMetric, ClassNursery):
    def __init__(self, datasets_manager: DatasetsManager):
        """

        Parameters
        ----------
        datasets_manager : DatasetsManager
            The dataset manager managing the labels and other information
        """
        super(PrecisionRecallFMeasure, self).__init__(datasets_manager=datasets_manager)
        self.datasets_manager = datasets_manager
        self.idx2labelname_mapping = None
        self.msg_printer = Printer()
        self.classification_metrics_utils = ClassificationMetricsUtils()
        self.label_namespace = self.datasets_manager.label_namespaces[0]
        self.normalized_probs_namespace = "normalized_probs"
        self.label_numericalizer = self.datasets_manager.namespace_to_numericalizer[
            self.label_namespace
        ]

        # setup counters to calculate true positives, false positives,
        # false negatives and true negatives
        # The keys are the different class indices in the dataset and the
        # values are the number of true positives, false positives, false negative
        # true negatvies for the dataset

        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def print_confusion_metrics(
        self,
        predicted_probs: torch.FloatTensor,
        labels: torch.LongTensor,
        labels_mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        """ Prints confusion matrix

        Parameters
        ----------
        predicted_probs : torch.FloatTensor
            Predicted Probabilities ``[batch_size, num_classes]``
        labels : torch.FloatTensor
            True labels of the size ``[batch_size, 1]``
        labels_mask : Optional[torch.ByteTensor]
            Labels mask indicating 1 in thos places where the true label is ignored
            Otherwise 0. It should be of same size as labels

        """
        assert predicted_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(predicted_probs.size())
        )

        assert labels.ndimension() == 2, self.msg_printer.fail(
            "The labels should have 2 dimension."
            "The labels that you passed have shape "
            "{0}".format(labels.size())
        )

        if labels_mask is None:
            labels_mask = torch.zeros_like(labels, dtype=torch.bool)

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = predicted_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.cpu().numpy().tolist()

        # convert labels to 1 dimension
        true_labels_numpy = labels.cpu().numpy().tolist()

        (
            confusion_mtrx,
            classes,
        ) = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            predicted_tag_indices=top_indices_numpy,
            true_tag_indices=true_labels_numpy,
            true_masked_label_indices=labels_mask,
        )

        if self.idx2labelname_mapping is not None:
            classes_with_names = [
                f"cls_{class_}({self.idx2labelname_mapping[class_]})"
                for class_ in classes
            ]
        else:
            classes_with_names = classes

        assert (
            len(classes) == confusion_mtrx.shape[1]
        ), f"len(classes) = {len(classes)} confusion matrix shape {confusion_mtrx.shape}"

        header = [f"{class_}" for class_ in classes]
        header.insert(0, "pred(cols)/true(rows)")

        confusion_mtrx = pd.DataFrame(confusion_mtrx)
        confusion_mtrx.insert(0, "class_name", classes_with_names)

        self.msg_printer.table(
            data=confusion_mtrx.values.tolist(), header=header, divider=True
        )

    def calc_metric(
        self, lines: List[Line], labels: List[Label], model_forward_dict: Dict[str, Any]
    ) -> None:
        """ Updates the values being tracked for calculating the metric

        For Precision Recall FMeasure we update the true positive,
        false positive and false negative of the different classes
        being tracked

        Parameters
        ----------
        lines : List[Line]
           A list of lines
        labels: List[Label]
            A list of labels. This has to be the label used for classification
            Refer to the documentation of Label for more information

        model_forward_dict : Dict[str, Any]
            The dictionary obtained after a forward pass
            The model_forward_pass is expected to have ``normalized_probs``
            that usually is of the size ``[batch_size, num_classes]``
        """

        normalized_probs = model_forward_dict[self.normalized_probs_namespace]

        labels_tensor = []
        for label in labels:
            tokens = label.tokens[self.label_namespace]
            tokens = [tok.text for tok in tokens]
            numericalized_instance = self.label_numericalizer.numericalize_instance(
                instance=tokens
            )

            labels_tensor.extend(numericalized_instance)

        labels_tensor = torch.LongTensor(labels_tensor)
        labels_tensor = labels_tensor.view(-1, 1)
        labels_mask = torch.zeros_like(labels_tensor).type(torch.ByteTensor)

        normalized_probs = normalized_probs.cpu()

        assert normalized_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(normalized_probs.size())
        )

        assert labels_tensor.ndimension() == 2, self.msg_printer.fail(
            "The labels should have 2 dimension."
            "The labels that you passed have shape "
            "{0}".format(labels_tensor.size())
        )

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = normalized_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.cpu().numpy().tolist()

        # convert labels to 1 dimension
        true_labels_numpy = labels_tensor.cpu().numpy().tolist()

        labels_mask = labels_mask.tolist()

        (
            confusion_mtrx,
            classes,
        ) = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            true_tag_indices=true_labels_numpy,
            predicted_tag_indices=top_indices_numpy,
            true_masked_label_indices=labels_mask,
        )

        # For further confirmation on how I calculated this I searched for stackoverflow on
        # 18th of July 2019. This seems to be the correct way to calculate tps, fps, fns
        # You can refer to https://stackoverflow.com/a/43331484/2704763

        # calculate tps
        tps = np.around(np.diag(confusion_mtrx), decimals=4)

        # calculate fps
        fps = np.around(np.sum(confusion_mtrx, axis=0) - tps, decimals=4)

        # calculate fns
        fns = np.around(np.sum(confusion_mtrx, axis=1) - tps, decimals=4)

        tps = tps.tolist()
        fps = fps.tolist()
        fns = fns.tolist()

        class_tps_mapping = dict(zip(classes, tps))
        class_fps_mapping = dict(zip(classes, fps))
        class_fns_mapping = dict(zip(classes, fns))

        self.tp_counter = merge_dictionaries_with_sum(
            self.tp_counter, class_tps_mapping
        )
        self.fp_counter = merge_dictionaries_with_sum(
            self.fp_counter, class_fps_mapping
        )
        self.fn_counter = merge_dictionaries_with_sum(
            self.fn_counter, class_fns_mapping
        )

    def get_metric(self) -> Dict[str, Any]:
        """ Returns different values being tracked to calculate Precision Recall FMeasure

        Returns
        -------
        Dict[str, Any]
            Returns a dictionary with the following key value pairs for every namespace

            precision: Dict[str, float]
                The precision for different classes
            recall: Dict[str, float]
                The recall values for different classes
            fscore: Dict[str, float]
                The fscore values for different classes,
            num_tp: Dict[str, int]
                The number of true positives for different classes,
            num_fp: Dict[str, int]
                The number of false positives for different classes,
            num_fn: Dict[str, int]
                The number of false negatives for different classes
            "macro_precision": float
                The macro precision value considering all different classes,
            macro_recall: float
                The macro recall value considering all different classes
            macro_fscore: float
                The macro fscore value considering all different classes
            micro_precision: float
                The micro precision value considering all different classes,
            micro_recall: float
                The micro recall value considering all different classes.
            micro_fscore: float
                The micro fscore value considering all different classes

        """
        (
            precision_dict,
            recall_dict,
            fscore_dict,
        ) = self.classification_metrics_utils.get_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        # for a detailed discussion on micro and macro scores please follow the discussion @
        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        # micro scores
        (
            micro_precision,
            micro_recall,
            micro_fscore,
        ) = self.classification_metrics_utils.get_micro_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        (
            macro_precision,
            macro_recall,
            macro_fscore,
        ) = self.classification_metrics_utils.get_macro_prf_from_prf_dicts(
            precision_dict=precision_dict,
            recall_dict=recall_dict,
            fscore_dict=fscore_dict,
        )

        metric = {
            self.label_namespace: {
                "precision": precision_dict,
                "recall": recall_dict,
                "fscore": fscore_dict,
                "num_tp": self.tp_counter,
                "num_fp": self.fp_counter,
                "num_fn": self.fn_counter,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_fscore": macro_fscore,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_fscore": micro_fscore,
            }
        }

        return metric

    def reset(self) -> None:
        """ Resets all the counters

        Resets the ``tp_counter`` which is the true positive counter
        Resets the ``fp_counter`` which is the false positive counter
        Resets the ``fn_counter`` - which is the false negative counter
        Resets the ``tn_counter`` - which is the true nagative counter

        """
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def report_metrics(self, report_type="wasabi"):
        """ Reports metrics in a printable format

        Parameters
        ----------
        report_type : type
            Select one of ``[wasabi, paper]``
            If wasabi, then we return a printable table that represents the
            precision recall and fmeasures for different classes

        """
        if report_type == "wasabi":
            table = self.classification_metrics_utils.generate_table_report_from_counters(
                tp_counter=self.tp_counter,
                fp_counter=self.fp_counter,
                fn_counter=self.fn_counter,
            )
            return {self.label_namespace: table}
