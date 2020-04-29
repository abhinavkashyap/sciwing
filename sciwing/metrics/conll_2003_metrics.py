from typing import Any, Dict, List

from sciwing.data.seq_label import SeqLabel
from sciwing.data.line import Line
from sciwing.data.token import Token
from sciwing.data.datasets_manager import DatasetsManager

from sciwing.metrics.BaseMetric import BaseMetric
import subprocess
import wasabi
from collections import defaultdict
import pathlib
import os
import numpy as np
import uuid


class ConLL2003Metrics(BaseMetric):
    """
    Returns the conll metrics for every namespace.
    The conll2003 metric assumes that the conlleval perl script is available
    It writes a file with true labels and pred labels for a namespace
    Parses the span level statistics which can then be used to select the model with the best
    F1 score
    """

    def __init__(
        self,
        datasets_manager: DatasetsManager,
        predicted_tags_namespace_prefix="predicted_tags",
        words_namespace: str = "tokens",
    ):
        super(ConLL2003Metrics, self).__init__(datasets_manager=datasets_manager)
        self.datasets_manager = datasets_manager
        self.label_namespaces = datasets_manager.label_namespaces
        self.words_namespace = words_namespace
        self.namespace_to_vocab = self.datasets_manager.namespace_to_vocab
        self.predicted_tags_namespace_prefix = predicted_tags_namespace_prefix
        self.msg_printer = wasabi.Printer()
        self.acc_counter: Dict[str, List[float]] = defaultdict(list)
        self.precision_counter: Dict[str, List[float]] = defaultdict(list)
        self.recall_counter: Dict[str, List[float]] = defaultdict(list)
        self.fmeasure_counter: Dict[str, List[float]] = defaultdict(list)

    def calc_metric(
        self,
        lines: List[Line],
        labels: List[SeqLabel],
        model_forward_dict: Dict[str, Any],
    ) -> None:

        line_tokens: List[List[Token]] = [line.tokens["tokens"] for line in lines]
        cwd = os.path.dirname(os.path.realpath(__file__))

        for namespace in self.label_namespaces:
            predicted_tags = model_forward_dict.get(
                f"{self.predicted_tags_namespace_prefix}_{namespace}"
            )

            true_labels = [label.tokens[namespace] for label in labels]
            namespace_filename = f"{cwd}/{str(uuid.uuid4())}_{namespace}_pred.txt"
            namespace_filename = pathlib.Path(namespace_filename)
            with open(namespace_filename, "w") as fp:

                for line_tokens_, true_labels_, predicted_tags_ in zip(
                    line_tokens, true_labels, predicted_tags
                ):
                    for line_token, true_label, predicted_tag in zip(
                        line_tokens_, true_labels_, predicted_tags_
                    ):
                        token_text = line_token.text
                        predicted_tag = self.namespace_to_vocab[
                            namespace
                        ].get_token_from_idx(predicted_tag)
                        true_label = true_label.text
                        fp.write(" ".join([token_text, true_label, predicted_tag]))
                        fp.write("\n")

            # invoke the perl script
            command = f"perl {cwd}/conlleval.perl < {str(namespace_filename)}"
            command = str(command)
            msg = "\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n"
            msg += "".join(os.popen(command).readlines())

            # parse the results
            accuracy_msg = msg.split("\n")[3]
            acc_prf = accuracy_msg.split(";")
            accuracy = acc_prf[0].split(":")[-1].strip().replace("%", "")
            precision = acc_prf[1].split(":")[-1].strip().replace("%", "")
            recall = acc_prf[2].split(":")[-1].strip().replace("%", "")
            fscore = acc_prf[3].split(":")[-1].strip().replace("%", "")

            # get in decimal points
            accuracy = float(accuracy) / 100
            precision = float(precision) / 100
            recall = float(recall) / 100
            fscore = float(fscore) / 100
            accuracy = np.round(accuracy, decimals=3)
            precision = np.round(precision, decimals=3)
            recall = np.round(recall, decimals=3)
            fscore = np.round(fscore, decimals=3)

            # update the counter
            self.acc_counter[namespace].append(accuracy)
            self.precision_counter[namespace].append(precision)
            self.recall_counter[namespace].append(recall)
            self.fmeasure_counter[namespace].append(fscore)

            # remove the file
            namespace_filename.unlink()

    def get_metric(self) -> Dict[str, Any]:
        metrics = {}
        for namespace in self.label_namespaces:
            acc = self.acc_counter[namespace]
            precision = self.precision_counter[namespace]
            recall = self.recall_counter[namespace]
            fscore = self.fmeasure_counter[namespace]

            acc = sum(acc) / len(acc)
            precision = sum(precision) / len(precision)
            recall = sum(recall) / len(recall)
            fscore = sum(fscore) / len(fscore)

            acc = np.round(acc, decimals=3)
            precision = np.round(precision, decimals=3)
            recall = np.round(recall, decimals=3)
            fscore = np.round(fscore, decimals=3)

            metrics[namespace] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
            }
        return metrics

    def report_metrics(self, report_type: str = "wasabi") -> Any:
        reports = {}
        if report_type == "wasabi":
            for namespace in self.label_namespaces:
                metric = self.get_metric()[namespace]
                acc = metric["accuracy"]
                precision = metric["precision"]
                recall = metric["recall"]
                fscore = metric["fscore"]

                # build table
                header_row = ["Metric", "Value"]
                rows = [
                    ("Acc", acc),
                    ("Precision", precision),
                    ("Recall", recall),
                    ("Fscore", fscore),
                ]

                table = wasabi.table(rows, header=header_row, divider=True)
                reports[namespace] = table

        return reports

    def reset(self):
        self.acc_counter = defaultdict(list)
        self.precision_counter = defaultdict(list)
        self.recall_counter = defaultdict(list)
        self.fmeasure_counter = defaultdict(list)
