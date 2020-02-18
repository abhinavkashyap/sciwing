from typing import Dict, Any, List, Optional, Union
import torch.nn as nn
from sciwing.infer.seq_label_inference.BaseSeqLabelInference import (
    BaseSeqLabelInference,
)
from sciwing.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from torch.utils.data import DataLoader
from sciwing.utils.tensor_utils import move_to_device
import torch
import pandas as pd
from sciwing.utils.vis_seq_tags import VisTagging
import wasabi
from deprecated import deprecated


@deprecated(
    reason="seq_label_inference which is more generic will be used for this"
    "and will be removed in version 0.1"
)
class ParscitInference(BaseSeqLabelInference):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        dataset: BaseSeqLabelingDataset,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):

        super(ParscitInference, self).__init__(
            model=model, model_filepath=model_filepath, dataset=dataset, device=device
        )

        self.msg_printer = wasabi.Printer()
        self.labelname2idx_mapping = self.dataset.get_classname2idx()
        self.idx2labelname_mapping = {
            idx: label_name for label_name, idx in self.labelname2idx_mapping.items()
        }
        self.metrics_calculator = TokenClassificationAccuracy(
            idx2labelname_mapping=self.idx2labelname_mapping
        )
        self.output_analytics = None
        self.output_df = None
        self.batch_size = 32
        self.load_model()

        num_categories = self.dataset.get_num_classes()
        categories = [self.idx2labelname_mapping[idx] for idx in range(num_categories)]
        self.seq_tagging_visualizer = VisTagging(tags=categories)

    def run_inference(self) -> Dict[str, Any]:
        loader = DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=False
        )
        output_analytics = {}
        sentences = []  # all the sentences that is seen till now
        predicted_tag_indices = []
        predicted_tag_names = []  # all the tags that are predicted for the sentences
        true_tag_indices = []
        true_tag_names = []

        for iter_dict in loader:
            iter_dict = move_to_device(iter_dict, cuda_device=self.device)
            model_output_dict = self.model_forward_on_iter_dict(iter_dict=iter_dict)
            self.metric_calc_on_iter_dict(
                iter_dict=iter_dict, model_output_dict=model_output_dict
            )
            batch_sentences = self.iter_dict_to_sentences(iter_dict=iter_dict)

            (
                predicted_tags,
                predicted_tag_strings,
            ) = self.model_output_dict_to_prediction_indices_names(
                model_output_dict=model_output_dict
            )
            true_tags, true_labels_strings = self.iter_dict_to_true_indices_names(
                iter_dict=iter_dict
            )

            sentences.extend(batch_sentences)

            predicted_tag_indices.extend(predicted_tags)
            predicted_tag_names.extend(predicted_tag_strings)
            true_tag_indices.extend(true_tags)
            true_tag_names.extend(true_labels_strings)

        output_analytics["true_tag_indices"] = true_tag_indices
        output_analytics["predicted_tag_indices"] = predicted_tag_indices
        output_analytics["true_tag_names"] = true_tag_names
        output_analytics["predicted_tag_names"] = predicted_tag_names
        output_analytics["sentences"] = sentences
        return output_analytics

    def print_confusion_matrix(self) -> None:
        """ Print confusion matrix for the test datasets
        """
        self.metrics_calculator.print_confusion_metrics(
            true_tag_indices=self.output_df["true_tag_indices"].tolist(),
            predicted_tag_indices=self.output_df["predicted_tag_indices"].tolist(),
        )

    def get_misclassified_sentences(
        self, first_class: int, second_class: int
    ) -> List[str]:
        """This returns the true label misclassified as
        pred label idx

        Parameters
        ----------
        first_class : int
            The label index of the true class name
        second_class : int
            The label index of the predicted class name


        Returns
        -------
        List[str]
            A list of strings where the true class is classified as pred class.

        """

        # get rows where true tag has first_class
        true_tag_indices = self.output_df.true_tag_indices.tolist()
        pred_tag_indices = self.output_df.predicted_tag_indices.tolist()

        indices = []

        for idx, (true_tag_index, pred_tag_index) in enumerate(
            zip(true_tag_indices, pred_tag_indices)
        ):
            true_tags_pred_tags = zip(true_tag_index, pred_tag_index)
            for true_tag, pred_tag in true_tags_pred_tags:
                if true_tag == first_class and pred_tag == second_class:
                    indices.append(idx)
                    break

        sentences = []

        for idx in indices:
            sentence = self.output_analytics["sentences"][idx].split()
            true_labels = self.output_analytics["true_tag_names"][idx].split()
            pred_labels = self.output_analytics["predicted_tag_names"][idx].split()
            len_sentence = len(sentence)
            true_labels = true_labels[:len_sentence]
            pred_labels = pred_labels[:len_sentence]
            stylized_string_true = self.seq_tagging_visualizer.visualize_tokens(
                sentence, true_labels
            )
            stylized_string_predicted = self.seq_tagging_visualizer.visualize_tokens(
                sentence, pred_labels
            )

            sentence = (
                f"GOLD LABELS \n{'*' * 80} \n{stylized_string_true} \n\n"
                f"PREDICTED LABELS \n{'*' * 80} \n{stylized_string_predicted}\n\n"
            )
            sentences.append(sentence)

        return sentences

    @deprecated(reason="Generate report for paper will be removed in version 0.2")
    def generate_report_for_paper(self):
        """ Generates just the fmeasures to be reported on paper
        """
        paper_report, row_names = self.metrics_calculator.report_metrics(
            report_type="paper"
        )
        return paper_report, row_names

    def model_forward_on_iter_dict(self, iter_dict: Dict[str, Any]):
        with torch.no_grad():
            model_output_dict = self.model(
                iter_dict, is_training=False, is_validation=False, is_test=True
            )
        return model_output_dict

    def metric_calc_on_iter_dict(
        self, iter_dict: Dict[str, Any], model_output_dict: Dict[str, Any]
    ):
        self.metrics_calculator.calc_metric(
            iter_dict=iter_dict, model_forward_dict=model_output_dict
        )

    def model_output_dict_to_prediction_indices_names(
        self, model_output_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        predicted_tags = model_output_dict["predicted_tags"]  # List[List[str]]
        predicted_tag_strings = []
        for predicted_tag in predicted_tags:
            pred_tag_string = self.dataset.get_class_names_from_indices(predicted_tag)
            pred_tag_string = " ".join(pred_tag_string)
            predicted_tag_strings.append(pred_tag_string)
        return predicted_tags, predicted_tag_strings

    def iter_dict_to_sentences(self, iter_dict: Dict[str, Any]):
        tokens = iter_dict["tokens"]
        tokens_list = tokens.tolist()
        batch_sentences = list(
            map(self.dataset.word_vocab.get_disp_sentence_from_indices, tokens_list)
        )
        return batch_sentences

    def iter_dict_to_true_indices_names(self, iter_dict: Dict[str, Any]):
        labels = iter_dict["label"]
        labels_list = labels.tolist()
        true_labels_strings = []
        for tags in labels_list:
            true_tag_names = self.dataset.get_class_names_from_indices(tags)
            true_tag_names = " ".join(true_tag_names)
            true_labels_strings.append(true_tag_names)

        true_labels_strings = list(true_labels_strings)
        return labels_list, true_labels_strings

    def infer_single_sentence(self, line: str) -> str:
        """ Return the tagged string for a single sentence

        Parameters
        ----------
        line : str
            A single sentence to be inferred

        Returns
        -------
        str
            Returns the tagged string for the line

        """
        len_words = len(line.split())
        iter_dict = self.dataset.get_iter_dict(line)
        iter_dict = move_to_device(iter_dict, cuda_device=self.device)
        iter_dict["tokens"] = iter_dict["tokens"].unsqueeze(0)
        iter_dict["char_tokens"] = iter_dict["char_tokens"].unsqueeze(0)

        model_output_dict = self.model_forward_on_iter_dict(iter_dict=iter_dict)
        _, predicted_tag_names = self.model_output_dict_to_prediction_indices_names(
            model_output_dict=model_output_dict
        )
        predicted_tag_names = predicted_tag_names[0].split()
        len_pred_tag_names = len(predicted_tag_names)
        infer_len = len_words if len_words < len_pred_tag_names else len_pred_tag_names
        predicted_tag_names = predicted_tag_names[:infer_len]
        predicted_tag_names = " ".join(predicted_tag_names)
        return predicted_tag_names

    def report_metrics(self):
        print(self.metrics_calculator.report_metrics())

    def run_test(self):
        self.output_analytics = self.run_inference()
        self.output_df = pd.DataFrame(self.output_analytics)
