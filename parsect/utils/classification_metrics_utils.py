import wasabi
import numpy as np
from typing import Dict, List, Optional
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class ClassificationMetricsUtils:
    """
    The classification metrics like accuracy, precision recall and fmeasure
    are often used in supervised learning. This class provides a few utilities
    that helps in calculating these.
    """

    def __init__(
        self,
        idx2labelname_mapping: Optional[Dict[int, str]] = None,
        masked_label_indices: Optional[List[int]] = None,
    ):
        self.msg_printer = wasabi.Printer()
        self.idx2labelname_mapping = idx2labelname_mapping
        self.mask_label_indices = masked_label_indices or []

    def get_prf_from_counters(
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

    def get_micro_prf_from_counters(
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
    def get_macro_prf_from_prf_dicts(
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

    def get_confusion_matrix_and_labels(
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

    def generate_table_report_from_counters(
        self,
        tp_counter: Dict[int, int],
        fp_counter: Dict[int, int],
        fn_counter: Dict[int, int],
    ):
        precision_dict, recall_dict, fscore_dict = self.get_prf_from_counters(
            tp_counter=tp_counter, fp_counter=fp_counter, fn_counter=fn_counter
        )
        micro_precision, micro_recall, micro_fscore = self.get_micro_prf_from_counters(
            tp_counter=tp_counter, fp_counter=fp_counter, fn_counter=fn_counter
        )
        macro_precision, macro_recall, macro_fscore = self.get_macro_prf_from_prf_dicts(
            precision_dict=precision_dict,
            recall_dict=recall_dict,
            fscore_dict=fscore_dict,
        )

        classes = precision_dict.keys()
        classes = sorted(classes)
        classes = filter(lambda class_: class_ not in self.mask_label_indices, classes)
        classes = list(classes)

        if self.idx2labelname_mapping is None:
            idx2labelname_mapping = {class_num: class_num for class_num in classes}
        else:
            idx2labelname_mapping = self.idx2labelname_mapping

        header_row = [" ", "Precision", "Recall", "F_measure"]
        rows = []
        for class_num in classes:
            p = precision_dict[class_num]
            r = recall_dict[class_num]
            f = fscore_dict[class_num]
            rows.append(
                (f"cls_{class_num} ({idx2labelname_mapping[int(class_num)]})", p, r, f)
            )

        rows.append(["-"] * 4)
        rows.append(["Macro", macro_precision, macro_recall, macro_fscore])
        rows.append(["Micro", micro_precision, micro_recall, micro_fscore])

        return wasabi.table(rows, header=header_row, divider=True)
