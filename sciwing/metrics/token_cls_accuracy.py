from typing import Dict, Union, Any, List, Optional
from sciwing.metrics.BaseMetric import BaseMetric
import wasabi
from sciwing.utils.common import merge_dictionaries_with_sum
import numpy as np
import pandas as pd
from sciwing.metrics.classification_metrics_utils import ClassificationMetricsUtils
import torch
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
from collections import defaultdict


class TokenClassificationAccuracy(BaseMetric, ClassNursery):
    def __init__(
        self,
        datasets_manager: DatasetsManager = None,
        predicted_tags_namespace_prefix="predicted_tags",
    ):
        super(TokenClassificationAccuracy, self).__init__(
            datasets_manager=datasets_manager
        )
        self.datasets_manager = datasets_manager
        self.label_namespaces = datasets_manager.label_namespaces
        self.predicted_tags_namespace_prefix = predicted_tags_namespace_prefix
        self.msg_printer = wasabi.Printer()
        self.classification_metrics_utils = ClassificationMetricsUtils()

        # a mapping between namespace and tp_counters for every class
        self.tp_counter: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.fp_counter: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.fn_counter: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.tn_counter: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def calc_metric(
        self,
        lines: List[Line],
        labels: List[SeqLabel],
        model_forward_dict: Dict[str, Any],
    ) -> None:
        """

        Parameters
        ----------------
        lines: List[Line]
            The list of lines

        labels: List[Label]
            The list of sequence labels

        model_forward_dict: Dict[str, Any]
            The model_forward_dict should have predicted tags for every namespace
            The predicted_tags are the best possible predicted tags for the batch
            They are List[List[int]] where the size is ``[batch_size, time_steps]``
            We expect that the predicted tags are

        """

        # get true labels for all namespaces
        namespace_to_true_labels = defaultdict(list)
        namespace_to_true_labels_mask = defaultdict(list)
        namespace_to_pred_labels_mask = defaultdict(list)

        for namespace in self.label_namespaces:
            # List[List[int]]
            predicted_tags = model_forward_dict.get(
                f"{self.predicted_tags_namespace_prefix}_{namespace}"
            )
            max_length = max([len(tags) for tags in predicted_tags])  # max num tokens
            numericalizer = self.datasets_manager.namespace_to_numericalizer[namespace]
            pred_tags_mask = numericalizer.get_mask_for_batch_instances(
                instances=predicted_tags
            ).tolist()
            namespace_to_pred_labels_mask[namespace] = pred_tags_mask

            for label in labels:
                true_labels = label.tokens[namespace]
                true_labels = [tok.text for tok in true_labels]

                true_labels = numericalizer.numericalize_instance(instance=true_labels)
                true_labels = numericalizer.pad_instance(
                    numericalized_text=true_labels,
                    max_length=max_length,
                    add_start_end_token=False,
                )
                labels_mask = numericalizer.get_mask_for_instance(
                    instance=true_labels
                ).tolist()
                namespace_to_true_labels[namespace].append(true_labels)
                namespace_to_true_labels_mask[namespace].append(labels_mask)

        for namespace in self.label_namespaces:
            labels_ = namespace_to_true_labels[namespace]
            labels_mask_ = namespace_to_true_labels_mask[namespace]
            pred_labels_mask_ = namespace_to_pred_labels_mask[namespace]
            # List[List[int]]
            predicted_tags = model_forward_dict.get(
                f"{self.predicted_tags_namespace_prefix}_{namespace}"
            )

            (
                confusion_mtrx,
                classes,
            ) = self.classification_metrics_utils.get_confusion_matrix_and_labels(
                true_tag_indices=labels_,
                predicted_tag_indices=predicted_tags,
                true_masked_label_indices=labels_mask_,
                pred_labels_mask=pred_labels_mask_,
            )

            tps = np.around(np.diag(confusion_mtrx), decimals=4)
            fps = np.around(np.sum(confusion_mtrx, axis=0) - tps, decimals=4)
            fns = np.around(np.sum(confusion_mtrx, axis=1) - tps, decimals=4)

            tps = tps.tolist()
            fps = fps.tolist()
            fns = fns.tolist()

            class_tps_mapping = dict(zip(classes, tps))
            class_fps_mapping = dict(zip(classes, fps))
            class_fns_mapping = dict(zip(classes, fns))

            self.tp_counter[namespace] = merge_dictionaries_with_sum(
                self.tp_counter.get(namespace, {}), class_tps_mapping
            )
            self.fp_counter[namespace] = merge_dictionaries_with_sum(
                self.fp_counter.get(namespace, {}), class_fps_mapping
            )
            self.fn_counter[namespace] = merge_dictionaries_with_sum(
                self.fn_counter.get(namespace, {}), class_fns_mapping
            )

    def get_metric(self) -> Dict[str, Union[Dict[str, float], float]]:
        """ Returns different values being tracked to calculate Precision Recall FMeasure
        Returns
        -------
        Dict[str, Any]
            Returns a dictionary with following key value pairs for every namespace
            precision: Dict[str, float]
                The precision for different classes
            recall: Dict[str, float]
                The recall values for different classes
            "fscore": Dict[str, float]
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

        metrics = {}

        for namespace in self.label_namespaces:
            (
                precision_dict,
                recall_dict,
                fscore_dict,
            ) = self.classification_metrics_utils.get_prf_from_counters(
                tp_counter=self.tp_counter[namespace],
                fp_counter=self.fp_counter[namespace],
                fn_counter=self.fn_counter[namespace],
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
                tp_counter=self.tp_counter[namespace],
                fp_counter=self.fp_counter[namespace],
                fn_counter=self.fn_counter[namespace],
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

            metrics[namespace] = {
                "precision": precision_dict,
                "recall": recall_dict,
                "fscore": fscore_dict,
                "num_tp": self.tp_counter[namespace],
                "num_fp": self.fp_counter[namespace],
                "num_fn": self.fn_counter[namespace],
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_fscore": macro_fscore,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_fscore": micro_fscore,
            }

        return metrics

    def report_metrics(self, report_type="wasabi") -> Any:
        """ Reports metrics in a printable format

       Parameters
       ----------
       report_type : type
           Select one of ``[wasabi, paper]``
           If wasabi, then we return a printable table that represents the
           precision recall and fmeasures for different classes

       """
        reports = {}
        for namespace in self.label_namespaces:
            if report_type == "wasabi":
                report = self.classification_metrics_utils.generate_table_report_from_counters(
                    tp_counter=self.tp_counter[namespace],
                    fp_counter=self.fp_counter[namespace],
                    fn_counter=self.fn_counter[namespace],
                    idx2labelname_mapping=self.datasets_manager.get_idx_label_mapping(
                        namespace
                    ),
                )
                reports[namespace] = report
        return reports

    def reset(self):
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def print_confusion_metrics(
        self,
        predicted_tag_indices: List[List[int]],
        true_tag_indices: List[List[int]],
        labels_mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        """ Prints confusion matrics for a batch of tag indices. It assumes that the batch
        is padded and every instance is of similar length

        Parameters
        ----------
        predicted_tag_indices : List[List[int]]
            Predicted tag indices for a batch of sentences
        true_tag_indices : List[List[int]]
            True tag indices for a batch of sentences
        labels_mask : Optional[torch.ByteTensor]
            The labels mask which has the same as ``true_tag_indices``.
            0 in a position indicates that there is no masking
            1 indicates that there is a masking

        """

        if labels_mask is None:
            labels_mask = torch.zeros_like(torch.Tensor(true_tag_indices)).type(
                torch.bool
            )

        (
            confusion_mtrx,
            classes,
        ) = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            predicted_tag_indices=predicted_tag_indices,
            true_tag_indices=true_tag_indices,
            true_masked_label_indices=labels_mask,
        )

        classes_with_names = classes

        confusion_mtrx = pd.DataFrame(confusion_mtrx)
        confusion_mtrx.insert(0, "class_name", classes_with_names)

        assert len(classes) == confusion_mtrx.shape[1] - 1

        header = [f"{class_}" for class_ in classes]
        header.insert(0, "pred(cols)/true(rows)")

        table = self.msg_printer.table(
            data=confusion_mtrx.values.tolist(), header=header, divider=True
        )
        print(table)
