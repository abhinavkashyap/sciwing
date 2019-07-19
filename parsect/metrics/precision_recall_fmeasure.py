import torch
from typing import Dict, Union, Any, Optional, List
from wasabi import Printer
from wasabi import table
from parsect.utils.common import merge_dictionaries_with_sum
import numpy as np
import pandas as pd
from parsect.metrics.BaseMetric import BaseMetric
from parsect.utils.classification_metrics_utils import ClassificationMetricsUtils


class PrecisionRecallFMeasure(BaseMetric):
    def __init__(
        self,
        idx2labelname_mapping: Dict[int, str],
        masked_label_indics: Optional[List[int]] = None,
    ):
        super(PrecisionRecallFMeasure, self).__init__()
        self.idx2labelname_mapping = idx2labelname_mapping
        self.mask_label_indices = masked_label_indics
        self.msg_printer = Printer()
        self.classification_metrics_utils = ClassificationMetricsUtils(
            masked_label_indices=self.mask_label_indices
        )

        # setup counters to calculate true positives, false positives,
        # false negatives and true negatives
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def print_confusion_metrics(
        self, predicted_probs: torch.FloatTensor, labels: torch.LongTensor
    ) -> None:
        assert predicted_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(predicted_probs.size())
        )

        assert labels.ndimension() == 2, self.msg_printer.fail(
            "The labels should have 1 dimension."
            "The labels that you passed have shape "
            "{0}".format(labels.size())
        )

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = predicted_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.cpu().numpy().tolist()

        # convert labels to 1 dimension
        true_labels_numpy = labels.cpu().numpy().tolist()

        confusion_mtrx, classes = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            true_labels_numpy, top_indices_numpy
        )
        classes_with_names = [
            f"cls_{class_}({self.idx2labelname_mapping[class_]})" for class_ in classes
        ]
        assert (
            len(classes) == confusion_mtrx.shape[1]
        ), f"len(classes) = {len(classes)} confusion matrix shape {confusion_mtrx.shape}"

        header = [f"{class_}" for class_ in classes]
        header.insert(0, "pred(cols)/true(rows)")
        print(header)

        confusion_mtrx = pd.DataFrame(confusion_mtrx)
        confusion_mtrx.insert(0, "class_name", classes_with_names)

        self.msg_printer.table(
            data=confusion_mtrx.values.tolist(), header=header, divider=True
        )

    def calc_metric(
        self, iter_dict: Dict[str, Any], model_forward_dict: Dict[str, Any]
    ) -> None:

        normalized_probs = model_forward_dict["normalized_probs"]
        labels = iter_dict["label"]
        normalized_probs = normalized_probs.cpu()
        labels = labels.cpu()

        assert normalized_probs.ndimension() == 2, self.msg_printer.fail(
            "The predicted probs should "
            "have 2 dimensions. The probs "
            "that you passed have shape "
            "{0}".format(normalized_probs.size())
        )

        assert labels.ndimension() == 2, self.msg_printer.fail(
            "The labels should have 1 dimension."
            "The labels that you passed have shape "
            "{0}".format(labels.size())
        )

        # TODO: for now k=1, change it to different number of ks
        top_probs, top_indices = normalized_probs.topk(k=1, dim=1)

        # convert to 1d numpy
        top_indices_numpy = top_indices.cpu().numpy().tolist()

        # convert labels to 1 dimension
        true_labels_numpy = labels.cpu().numpy().tolist()

        confusion_mtrx, classes = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            true_labels_numpy, top_indices_numpy
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

    def get_metric(self) -> Dict[str, Union[Dict[str, float], float]]:
        precision_dict, recall_dict, fscore_dict = self.classification_metrics_utils.get_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        # for a detailed discussion on micro and macro scores please follow the discussion @
        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        # micro scores
        micro_precision, micro_recall, micro_fscore = self.classification_metrics_utils.get_micro_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        macro_precision, macro_recall, macro_fscore = self.classification_metrics_utils.get_macro_prf_from_prf_dicts(
            precision_dict=precision_dict,
            recall_dict=recall_dict,
            fscore_dict=fscore_dict,
        )

        return {
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

    def reset(self) -> None:
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def report_metrics(self, report_type="wasabi"):

        accuracy_metrics = self.get_metric()
        precision = accuracy_metrics["precision"]
        recall = accuracy_metrics["recall"]
        fscore = accuracy_metrics["fscore"]
        macro_precision = accuracy_metrics["macro_precision"]
        macro_recall = accuracy_metrics["macro_recall"]
        macro_fscore = accuracy_metrics["macro_fscore"]
        micro_precision = accuracy_metrics["micro_precision"]
        micro_recall = accuracy_metrics["micro_recall"]
        micro_fscore = accuracy_metrics["micro_fscore"]

        if report_type == "wasabi":
            classes = precision.keys()
            classes = sorted(classes)
            header_row = [" ", "Precision", "Recall", "F_measure"]
            rows = []
            for class_num in classes:
                p = precision[class_num]
                r = recall[class_num]
                f = fscore[class_num]
                rows.append(
                    (
                        f"cls_{class_num} ({self.idx2labelname_mapping[int(class_num)]})",
                        p,
                        r,
                        f,
                    )
                )

            rows.append(["-"] * 4)
            rows.append(["Macro", macro_precision, macro_recall, macro_fscore])
            rows.append(["Micro", micro_precision, micro_recall, micro_fscore])

            return table(rows, header=header_row, divider=True)

        elif report_type == "paper":
            "Refer to the paper Logical Structure Recovery in Scholarly Articles with " "Rich Document Features Table 2. It generates just fscores and returns"
            class_nums = fscore.keys()
            class_nums = sorted(class_nums, reverse=False)
            fscores = [fscore[class_num] for class_num in class_nums]
            fscores.extend([micro_fscore, macro_fscore])
            return fscores
