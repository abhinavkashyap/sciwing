import sciwing.constants as constants
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from sciwing.infer.classification.BaseClassificationInference import (
    BaseClassificationInference,
)
from sciwing.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sciwing.utils.tensor_utils import move_to_device
from wasabi.util import MESSAGES

FILES = constants.FILES

SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ClassificationInference(BaseClassificationInference):
    """
    The sciwing engine runs the test lines through the classifier
    and returns the predictions/probabilities for different classes
    At a later point in time this method should be able to take any
    context of lines (may be from a file) and produce the output.

    This class also helps in performing various interactions with
    the results on the test dataset.
    Some features are
    1) Show confusion matrix
    2) Investigate a particular example in the test dataset
    3) Get instances that were classified as 2 when their true label is 1 and others

    All it needs is the configuration file stored under every experiment to have a
    vocab already stored in the experiment folder
    """

    def __init__(
        self, model: nn.Module, model_filepath: str, dataset: BaseTextClassification
    ):

        super(ClassificationInference, self).__init__(
            model=model, model_filepath=model_filepath, dataset=dataset
        )
        self.batch_size = 32

        self.labelname2idx_mapping = self.dataset.get_classname2idx()
        self.idx2labelname_mapping = {
            idx: label_name for label_name, idx in self.labelname2idx_mapping.items()
        }
        self.load_model()
        self.metrics_calculator = PrecisionRecallFMeasure(
            idx2labelname_mapping=self.idx2labelname_mapping
        )
        self.output_analytics = None

        # create a dataframe with all the information
        self.output_df = None

    def run_inference(self) -> Dict[str, Any]:
        loader = DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=False
        )
        output_analytics = {}

        # contains the predicted class names for all the instances
        pred_class_names = []
        true_class_names = []  # contains the true class names for all the instances
        sentences = []  # batch sentences in english
        true_labels_indices = []
        predicted_labels_indices = []
        all_pred_probs = []

        for iter_dict in loader:
            iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
            batch_sentences = self.iter_dict_to_sentences(iter_dict)
            model_output_dict = self.model_forward_on_iter_dict(iter_dict)
            normalized_probs = model_output_dict["normalized_probs"]
            self.metrics_calculator.calc_metric(
                iter_dict=iter_dict, model_forward_dict=model_output_dict
            )
            true_label_ind, true_label_names = self.iter_dict_to_true_indices_names(
                iter_dict=iter_dict
            )
            pred_label_indices, pred_label_names = self.model_output_dict_to_prediction_indices_names(
                model_output_dict=model_output_dict
            )

            true_label_ind = torch.LongTensor(true_label_ind)
            true_labels_indices.append(true_label_ind)
            true_class_names.extend(true_label_names)
            predicted_labels_indices.extend(pred_label_indices)
            pred_class_names.extend(pred_label_names)
            sentences.extend(batch_sentences)
            all_pred_probs.append(normalized_probs)

        # contains predicted probs for all the instances
        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        true_labels_indices = torch.cat(true_labels_indices, dim=0).squeeze()

        # torch.LongTensor N, 1
        output_analytics["true_labels_indices"] = true_labels_indices
        output_analytics["predicted_labels_indices"] = predicted_labels_indices
        output_analytics["pred_class_names"] = pred_class_names
        output_analytics["true_class_names"] = true_class_names
        output_analytics["sentences"] = sentences
        output_analytics["all_pred_probs"] = all_pred_probs

        return output_analytics

    def model_forward_on_iter_dict(self, iter_dict: Dict[str, Any]):
        with torch.no_grad():
            model_output_dict = self.model(
                iter_dict, is_training=False, is_validation=False, is_test=True
            )
        return model_output_dict

    def get_misclassified_sentences(
        self, true_label_idx: int, pred_label_idx: int
    ) -> List[str]:
        """
        This returns the true label misclassified as
        pred label idx
        :param true_label_idx: type: int
        :param pred_label_idx: type: int
        """
        instances_idx = self.output_df[
            self.output_df["true_labels_indices"].isin([true_label_idx])
            & self.output_df["predicted_labels_indices"].isin([pred_label_idx])
        ].index.tolist()

        sentences = []
        for idx in instances_idx:
            sentence = self.output_analytics["sentences"][idx]

            if true_label_idx != pred_label_idx:
                stylized_sentence = self.msg_printer.text(
                    title=sentence,
                    icon=MESSAGES.FAIL,
                    color=MESSAGES.FAIL,
                    no_print=True,
                )
            else:
                stylized_sentence = self.msg_printer.text(
                    title=sentence,
                    icon=MESSAGES.GOOD,
                    color=MESSAGES.GOOD,
                    no_print=True,
                )

            sentences.append(stylized_sentence)

        return sentences

    def print_confusion_matrix(self) -> None:
        self.metrics_calculator.print_confusion_metrics(
            predicted_probs=self.output_analytics["all_pred_probs"],
            labels=self.output_analytics["true_labels_indices"].unsqueeze(1),
        )

    def print_metrics(self):
        prf_table = self.metrics_calculator.report_metrics()
        print(prf_table)

    def generate_report_for_paper(self):
        paper_report = self.metrics_calculator.report_metrics(report_type="paper")
        class_numbers = sorted(self.idx2labelname_mapping.keys(), reverse=False)
        row_names = [
            f"class_{class_num} - ({self.idx2labelname_mapping[class_num]})"
            for class_num in class_numbers
        ]
        row_names.extend([f"Micro-Fscore", f"Macro-Fscore"])
        return paper_report, row_names

    def metric_calc_on_iter_dict(
        self, iter_dict: Dict[str, Any], model_output_dict: Dict[str, Any]
    ):
        self.metrics_calculator.calc_metric(
            iter_dict=iter_dict, model_forward_dict=model_output_dict
        )

    def model_output_dict_to_prediction_indices_names(
        self, model_output_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        normalized_probs = model_output_dict["normalized_probs"]
        pred_probs, pred_indices = torch.topk(normalized_probs, k=1, dim=1)
        pred_indices = pred_indices.squeeze(1).tolist()
        pred_classnames = [
            self.idx2labelname_mapping[pred_index] for pred_index in pred_indices
        ]
        return pred_indices, pred_classnames

    def infer_batch(self, lines: List[str]):
        iter_dict = self.dataset.get_iter_dict(lines=lines)

        if len(lines) == 1:
            iter_dict["tokens"] = iter_dict["tokens"].unsqueeze(0)

        model_output_dict = self.model_forward_on_iter_dict(iter_dict=iter_dict)
        _, pred_classnames = self.model_output_dict_to_prediction_indices_names(
            model_output_dict=model_output_dict
        )
        return pred_classnames

    def on_user_input(self, line: str) -> str:
        return self.infer_batch(lines=[line])[0]

    def iter_dict_to_sentences(self, iter_dict: Dict[str, Any]):
        tokens = iter_dict["tokens"]  # N * max_length
        tokens_list = tokens.tolist()
        batch_sentences = list(
            map(self.dataset.word_vocab.get_disp_sentence_from_indices, tokens_list)
        )
        return batch_sentences

    def iter_dict_to_true_indices_names(
        self, iter_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        labels = iter_dict["label"]  # N, 1
        labels_list = labels.squeeze().tolist()
        true_label_names = self.dataset.get_class_names_from_indices(labels_list)
        return labels_list, true_label_names

    def run_test(self):
        self.output_analytics = self.run_inference()
        self.output_df = pd.DataFrame(self.output_analytics)
