from typing import Dict, Union, Any, List
from parsect.metrics.BaseMetric import BaseMetric
import wasabi
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from parsect.utils.common import merge_dictionaries_with_sum
import numpy as np
import pandas as pd


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
        confusion_mtrx, classes = self._get_confusion_matrix_and_labels(
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

        precision_dict, recall_dict, fscore_dict = self._get_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        # for a detailed discussion on micro and macro scores please follow the discussion @
        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        # micro scores
        micro_precision, micro_recall, micro_fscore = self._get_micro_prf_from_counters(
            tp_counter=self.tp_counter,
            fp_counter=self.fp_counter,
            fn_counter=self.fn_counter,
        )

        # macro scores
        macro_precision, macro_recall, macro_fscore = self._get_macro_prf_from_prf_dicts(
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

        confusion_mtrx, classes = self._get_confusion_matrix_and_labels(
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

    def _get_confusion_matrix_and_labels(
        self, predicted_tag_indices: List[List[int]], true_tag_indices: List[List[int]]
    ) -> (np.array, List[int]):
        """
        Returns the confusion matrix and associated classes in the order
        :param predicted_tag_indices:
        :param true_tag_indices:
        :return:
        """
        predicted_tags_flat = list(itertools.chain.from_iterable(predicted_tag_indices))
        labels = list(itertools.chain.from_iterable(true_tag_indices))
        predicted_tags_flat = np.array(predicted_tags_flat)
        labels_numpy = np.array(labels)

        # filter the classes
        classes = unique_labels(labels_numpy, predicted_tags_flat)
        classes = filter(lambda class_: class_ not in self.mask_label_indices, classes)
        classes = list(classes)

        confusion_mtrx = confusion_matrix(
            labels_numpy, predicted_tags_flat, labels=classes
        )
        return confusion_mtrx, classes

    def _get_prf_from_counters(
        self,
        tp_counter: Dict[int, int],
        fp_counter: Dict[int, int],
        fn_counter: Dict[int, int],
    ):
        """
        Pass the dictionary that captures the class -> number of tps mapping
        and similarly other dictionaries that capture the class -> fp and fn mapping
        From This we calculate the
        :param tp_counter: type: Dict[int, int]
        :param fp_counter: type: Dict[int, int]
        :param fn_counter: type: Dict[int, int]
        :return: Dict[int, int], Dict[int, int], Dict[int, int]
        Three dictionaries representing the Precision Recall and Fmeasure
        for all the different classes
        """
        tp_classes = tp_counter.keys()
        fp_classes = fp_counter.keys()
        fn_classes = fn_counter.keys()
        assert tp_classes == fp_classes == fn_classes
        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}

        for class_ in tp_counter.keys():
            tp = tp_counter[class_]
            fp = fp_counter[class_]
            fn = fn_counter[class_]
            if tp == 0 and fp == 0:
                precision = 0
                self.msg_printer.warn("both tp and fp are 0 .. setting precision to 0")
            else:
                precision = tp / (tp + fp)
                precision = np.around(precision, decimals=4)
            if tp == 0 and fn == 0:
                recall = 0
                self.msg_printer.warn("both tp and fn are 0 .. setting recall to 0")
            else:
                recall = tp / (tp + fn)
                recall = np.around(recall, decimals=4)

            if precision == 0 and recall == 0:
                fscore = 0
                self.msg_printer.warn(
                    "both precision and recall are 0 .. setting fscore to 0"
                )
            else:
                fscore = (2 * precision * recall) / (precision + recall)
                fscore = np.around(fscore, decimals=4)

            precision_dict[class_] = precision
            recall_dict[class_] = recall
            fscore_dict[class_] = fscore
        return precision_dict, recall_dict, fscore_dict

    def _get_micro_prf_from_counters(
        self,
        tp_counter: Dict[int, int],
        fp_counter: Dict[int, int],
        fn_counter: Dict[int, int],
    ) -> (int, int, int):
        # micro scores
        all_num_tps = [num_tp for num_tp in tp_counter.values()]
        all_num_fps = [num_fp for num_fp in fp_counter.values()]
        all_num_fns = [num_fn for num_fn in fn_counter.values()]

        if np.sum(all_num_tps + all_num_fps) == 0:
            micro_precision = 0
            self.msg_printer.warn("Micro Precision is being set to 0")
        else:
            micro_precision = np.sum(all_num_tps) / np.sum(all_num_tps + all_num_fps)
            micro_precision = np.around(micro_precision, decimals=4)

        if np.sum(all_num_tps + all_num_fns) == 0:
            micro_recall = 0
            self.msg_printer.warn("Micro Recall is being set to 0")
        else:
            micro_recall = np.sum(all_num_tps) / np.sum(all_num_tps + all_num_fns)
            micro_recall = np.around(micro_recall, decimals=4)

        if micro_precision == 0 and micro_recall == 0:
            micro_fscore = 0.0
            self.msg_printer.warn("Micro Fscore is being set to 0")
        else:
            micro_fscore = (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            )
            micro_fscore = np.around(micro_fscore, decimals=4)

        return micro_precision, micro_recall, micro_fscore

    @staticmethod
    def _get_macro_prf_from_prf_dicts(
        precision_dict: Dict[int, int],
        recall_dict: Dict[int, int],
        fscore_dict: Dict[int, int],
    ) -> (int, int, int):
        all_precisions = [
            precision_value for precision_value in precision_dict.values()
        ]
        all_recalls = [recall_value for recall_value in recall_dict.values()]
        all_fscores = [fscores_value for fscores_value in fscore_dict.values()]

        # macro scores
        macro_precision = np.mean(all_precisions)
        macro_recall = np.mean(all_recalls)
        macro_fscore = np.mean(all_fscores)
        macro_precision = np.around(macro_precision, decimals=4)
        macro_recall = np.around(macro_recall, decimals=4)
        macro_fscore = np.around(macro_fscore, decimals=4)

        return macro_precision, macro_recall, macro_fscore
