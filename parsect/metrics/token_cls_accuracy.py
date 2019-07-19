from typing import Dict, Union, Any, List
from parsect.metrics.BaseMetric import BaseMetric
import wasabi
from parsect.utils.common import merge_dictionaries_with_sum
import numpy as np
import pandas as pd
from parsect.utils.classification_metrics_utils import ClassificationMetricsUtils


class TokenClassificationAccuracy(BaseMetric):
    def __init__(
        self,
        idx2labelname_mapping: Dict[int, str],
        mask_label_indices: Union[List[int], None] = None,
    ):
        super(TokenClassificationAccuracy, self).__init__()
        self.idx2labelname_mapping = idx2labelname_mapping
        self.mask_label_indices = mask_label_indices or []
        self.msg_printer = wasabi.Printer()
        self.classification_metrics_utils = ClassificationMetricsUtils(
            idx2labelname_mapping=idx2labelname_mapping,
            masked_label_indices=self.mask_label_indices,
        )

        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def calc_metric(
        self, iter_dict: Dict[str, Any], model_forward_dict: Dict[str, Any]
    ) -> None:
        """
        The iter_dict should have label key
        The label are gold labels for the batch
        They should have the shape batch_size, time_steps
        where time_steps are the size of the sequence

        The model_forward_dict should have predicted tags key
        The predicted tags are the best possible predicted tags for the batch
        They are List[List[int]] where the size is batch_size, time_steps

        :param iter_dict: Dict[str, Any]
        :param model_forward_dict: Dict[str, Any]
        :return: None
        """
        labels = iter_dict.get("label", None)
        labels = labels.cpu()
        predicted_tags = model_forward_dict.get(
            "predicted_tags", None
        )  # List[List[int]]

        if labels is None or predicted_tags is None:
            raise ValueError(
                f"While calling {self.__class__.__name__}, the iter_dict should"
                f"have a key called label and model_forward_dict "
                f"should have predicted_tags"
            )

        assert labels.ndimension() == 2, self.msg_printer.fail(
            f"The labels  for the metric {self.__class__.__name__} should have 2 dimensions."
            f"The labels that you passed have the shape {labels.size()}"
        )

        # flatten predicted tags to a single dimension
        confusion_mtrx, classes = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            true_tag_indices=labels.numpy().tolist(),
            predicted_tag_indices=predicted_tags,
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

    def report_metrics(self, report_type="wasabi") -> Any:
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

            return wasabi.table(rows, header=header_row, divider=True)

        elif report_type == "paper":
            class_nums = fscore.keys()
            class_nums = sorted(class_nums, reverse=False)
            fscores = [fscore[class_num] for class_num in class_nums]
            fscores.extend([micro_fscore, macro_fscore])
            rownames = [
                f"class_{class_num} - ({self.idx2labelname_mapping[class_num]})"
                for class_num in class_nums
            ]
            rownames.extend(["Micro-Fscore", "Macro-Fscore"])
            assert len(rownames) == len(fscores)
            return fscores, rownames

    def reset(self):
        self.tp_counter = {}
        self.fp_counter = {}
        self.fn_counter = {}
        self.tn_counter = {}

    def print_confusion_metrics(
        self, predicted_tag_indices: List[List[int]], true_tag_indices: List[List[int]]
    ) -> None:

        confusion_mtrx, classes = self.classification_metrics_utils.get_confusion_matrix_and_labels(
            predicted_tag_indices=predicted_tag_indices,
            true_tag_indices=true_tag_indices,
        )
        classes_with_names = [
            f"cls_{class_}({self.idx2labelname_mapping[class_]})" for class_ in classes
        ]

        confusion_mtrx = pd.DataFrame(confusion_mtrx)
        confusion_mtrx.insert(0, "class_name", classes_with_names)

        assert len(classes) == confusion_mtrx.shape[1] - 1

        header = [f"{class_}" for class_ in classes]
        header.insert(0, "pred(cols)/true(rows)")

        table = self.msg_printer.table(
            data=confusion_mtrx.values.tolist(), header=header, divider=True
        )
        print(table)
