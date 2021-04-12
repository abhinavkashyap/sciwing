from typing import Any, Dict, List

from sciwing.data.seq_label import SeqLabel
from sciwing.data.line import Line
from sciwing.data.token import Token
from sciwing.data.datasets_manager import DatasetsManager

from sciwing.metrics.BaseMetric import BaseMetric
import subprocess
import wasabi
from collections import defaultdict, Counter
import pathlib
import os
import numpy as np
import uuid


class SummarizationMetrics(BaseMetric):
    """
    Returns rouge for every namespace.
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
        super(SummarizationMetrics, self).__init__(datasets_manager=datasets_manager)
        self.datasets_manager = datasets_manager
        self.label_namespaces = datasets_manager.label_namespaces
        self.words_namespace = words_namespace
        self.namespace_to_vocab = self.datasets_manager.namespace_to_vocab
        self.predicted_tags_namespace_prefix = predicted_tags_namespace_prefix
        self.msg_printer = wasabi.Printer()

        self.rouge_1_counter: Dict[str, List[float]] = defaultdict(list)
        self.rouge_2_counter: Dict[str, List[float]] = defaultdict(list)
        self.rouge_l_counter: Dict[str, List[float]] = defaultdict(list)

    def calc_metric(
        self, lines: List[Line], labels: List[Line], model_forward_dict: Dict[str, Any]
    ) -> None:

        # line_tokens: List[List[Token]] = [line.tokens["tokens"] for line in lines]
        # true_label_text = [label.text for label in labels]
        cwd = os.path.dirname(os.path.realpath(__file__))

        for namespace in [self.words_namespace]:
            predicted_tags = model_forward_dict.get(
                f"{self.predicted_tags_namespace_prefix}_{namespace}"
            )

            true_summary_tokens: List[List[Token]] = [
                summary.tokens[namespace] for summary in labels
            ]
            true_summary_token_strs: List[List[str]] = [
                [token.text for token in tokens] for tokens in true_summary_tokens
            ]

            namespace_filename = f"{cwd}/{str(uuid.uuid4())}_{namespace}_pred.txt"
            namespace_filename = pathlib.Path(namespace_filename)

            predicted_summary_token_strs = []

            with open(namespace_filename, "w") as fp:
                for line, true_summary_token_strs_, predicted_tags_ in zip(
                    lines, true_summary_token_strs, predicted_tags
                ):
                    predicted_summary_token_strs_ = []

                    for predicted_tag in predicted_tags_:
                        predicted_tag = self.namespace_to_vocab[
                            namespace
                        ].get_token_from_idx(predicted_tag)
                        predicted_summary_token_strs_.append(predicted_tag)
                    predicted_summary_token_strs.append(predicted_summary_token_strs_)

                    fp.write(line.text)
                    fp.write("Ground Truth")
                    fp.write(
                        " ".join([f'"{token}"' for token in true_summary_token_strs_])
                    )
                    fp.write("Predicted")
                    fp.write(
                        " ".join(
                            [f'"{token}"' for token in predicted_summary_token_strs_]
                        )
                    )
                    fp.write("\n")

            for true_summary_token_strs_, predicted_summary_token_strs_ in zip(
                true_summary_token_strs, predicted_summary_token_strs
            ):
                rouge_1 = self._rouge_n(
                    predicted_summary_token_strs_, true_summary_token_strs_, 1
                )
                rouge_2 = self._rouge_n(
                    predicted_summary_token_strs_, true_summary_token_strs_, 2
                )
                rouge_l = self._rouge_l(
                    predicted_summary_token_strs_, true_summary_token_strs_
                )

                rouge_1 = np.round(rouge_1, decimals=3)
                rouge_2 = np.round(rouge_2, decimals=3)
                rouge_l = np.round(rouge_l, decimals=3)

                # update the counter
                self.rouge_1_counter[namespace].append(rouge_1)
                self.rouge_2_counter[namespace].append(rouge_2)
                self.rouge_l_counter[namespace].append(rouge_l)

    def get_metric(self) -> Dict[str, Any]:
        metrics = {}
        for namespace in [self.words_namespace]:
            rouge_1s = self.rouge_1_counter[namespace]
            rouge_2s = self.rouge_2_counter[namespace]
            rouge_ls = self.rouge_l_counter[namespace]

            rouge_1 = sum(rouge_1s) / len(rouge_1s)
            rouge_2 = sum(rouge_2s) / len(rouge_2s)
            rouge_l = sum(rouge_ls) / len(rouge_ls)

            rouge_1 = np.round(rouge_1, decimals=3)
            rouge_2 = np.round(rouge_2, decimals=3)
            rouge_l = np.round(rouge_l, decimals=3)

            metrics[namespace] = {
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
            }
        return metrics

    def report_metrics(self, report_type: str = "wasabi") -> Any:
        reports = {}
        if report_type == "wasabi":
            for namespace in [self.words_namespace]:
                metric = self.get_metric()[namespace]
                rouge_1 = metric["rouge_1"]
                rouge_2 = metric["rouge_2"]
                rouge_l = metric["rouge_l"]

                # build table
                header_row = ["Metric", "Value"]
                rows = [
                    ("Rouge_1", rouge_1),
                    ("Rouge_2", rouge_2),
                    ("Rouge_l", rouge_l),
                ]

                table = wasabi.table(rows, header=header_row, divider=True)
                reports[namespace] = table

        return reports

    def reset(self):
        self.rouge_1_counter = defaultdict(list)
        self.rouge_2_counter = defaultdict(list)
        self.rouge_l_counter = defaultdict(list)

    def _calc_f1(self, matches, count_for_recall, count_for_precision, alpha):
        def safe_div(x1, x2):
            return 0 if x2 == 0 else x1 / x2

        recall = safe_div(matches, count_for_recall)
        precision = safe_div(matches, count_for_precision)
        denom = (1.0 - alpha) * precision + alpha * recall
        return safe_div(precision * recall, denom)

    def _lcs(self, a, b):
        longer = a
        base = b
        if len(longer) < len(base):
            longer, base = base, longer

        if len(base) == 0:
            return 0

        row = [0] * len(base)
        for c_a in longer:
            left = 0
            upper_left = 0
            for i, c_b in enumerate(base):
                up = row[i]
                if c_a == c_b:
                    value = upper_left + 1
                else:
                    value = max(left, up)
                row[i] = value
                left = value
                upper_left = up

        return left

    def _len_ngram(self, words, n):
        return max(len(words) - n + 1, 0)

    def _ngram_iter(self, words, n):
        for i in range(self._len_ngram(words, n)):
            n_gram = words[i : i + n]
            yield tuple(n_gram)

    def _count_ngrams(self, words, n):
        c = Counter(self._ngram_iter(words, n))
        return c

    def _count_overlap(self, summary_ngrams, reference_ngrams):
        result = 0
        for k, v in summary_ngrams.items():
            result += min(v, reference_ngrams[k])
        return result

    def _rouge_n(self, pred_summary, true_summary, n, alpha=0.5):
        """
        Calculate ROUGE-N score.
        Parameters
        ----------
        pred_summary: list of list of str
            generated summary after tokenization
        true_summary: list of list of str
            reference or references to evaluate summary
        n: int
            ROUGE kind. n=1, calculate when ROUGE-1
        alpha: float (0~1)
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))
        Returns
        -------
        f1: float
            f1 score
        """
        pred_ngrams = self._count_ngrams(pred_summary, n)
        r_ngrams = self._count_ngrams(true_summary, n)
        matches = self._count_overlap(pred_ngrams, r_ngrams)
        count_for_recall = self._len_ngram(true_summary, n)
        count_for_prec = self._len_ngram(pred_summary, n)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1

    def _rouge_l(self, pred_summary, true_summary, alpha=0.5):
        """
        Calculate ROUGE-L score.
        Parameters
        ----------
        pred_summary: list of list of str
            generated summary after tokenization
        true_summary: list of list of str
            reference or references to evaluate summary
        n: int
            ROUGE kind. n=1, calculate when ROUGE-1
        alpha: float (0~1)
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))

        Returns
        -------
        f1: float
            f1 score
        """
        matches = self._lcs(true_summary, pred_summary)
        count_for_recall = len(true_summary)
        count_for_prec = len(pred_summary)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1
