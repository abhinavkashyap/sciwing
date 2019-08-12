import parsect.constants as constants
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from parsect.infer.classification.BaseClassificationInference import (
    BaseClassificationInference,
)
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from parsect.utils.tensor_utils import move_to_device
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from wasabi.util import MESSAGES
import json

FILES = constants.FILES

SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ParsectInference(BaseClassificationInference):
    """
    The parsect engine runs the test lines through the classifier
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
        self,
        model: nn.Module,
        model_filepath: str,
        hyperparam_config_filepath: str,
        dataset_class: type,
        dataset: Optional[BaseTextClassification] = None,
    ):
        """
        :param model: type: torch.nn.Module
        Pass the model on which inference should be run
        :param model_filepath: type: str
        The model filepath is the chkpoint file where the model state is stored
        :param hyperparam_config_filepath: type: str
        The path where all hyper-parameters necessary for restoring the model
        is necessary
        """
        super(ParsectInference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            hyperparam_config_filepath=hyperparam_config_filepath,
            dataset=dataset,
        )
        self.dataset_class = dataset_class

        with open(hyperparam_config_filepath, "r") as fp:
            config = json.load(fp)

        self.vocab_store_location = config.get("VOCAB_STORE_LOCATION", None)
        self.max_length = config.get("MAX_LENGTH", None) or 200
        self.batch_size = config.get("BATCH_SIZE", None) or 32

        if self.test_dataset is not None:
            self.labelname2idx_mapping = self.test_dataset.get_classname2idx()
            self.idx2labelname_mapping = {
                idx: label_name
                for label_name, idx in self.labelname2idx_mapping.items()
            }
            self.metrics_calculator = PrecisionRecallFMeasure(
                idx2labelname_mapping=self.idx2labelname_mapping
            )
            self.output_analytics = self.run_inference()

            # create a dataframe with all the information
            self.output_df = pd.DataFrame(self.output_analytics)

        self.load_model()

    def run_inference(self) -> Dict[str, Any]:
        loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        output_analytics = {}
        pred_class_names = (
            []
        )  # contains the predicted class names for all the instances
        true_class_names = []  # contains the true class names for all the instances
        sentences = []  # batch sentences in english
        true_labels_indices = []
        predicted_labels_indices = []
        all_pred_probs = []

        for iter_dict in loader:
            iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
            tokens = iter_dict["tokens"]  # N * max_length
            labels = iter_dict["label"]  # N, 1
            labels_list = labels.squeeze().tolist()
            tokens_list = tokens.tolist()

            batch_sentences = list(
                map(self.test_dataset.get_disp_sentence_from_indices, tokens_list)
            )

            with torch.no_grad():
                model_output_dict = self.model(
                    iter_dict, is_training=False, is_validation=False, is_test=True
                )
            normalized_probs = model_output_dict["normalized_probs"]
            self.metrics_calculator.calc_metric(
                iter_dict=iter_dict, model_forward_dict=model_output_dict
            )

            top_probs, top_indices = torch.topk(normalized_probs, k=1, dim=1)
            top_indices_list = top_indices.squeeze().tolist()

            pred_label_names = self.test_dataset.get_class_names_from_indices(
                top_indices_list
            )
            true_label_names = self.test_dataset.get_class_names_from_indices(
                labels_list
            )

            true_labels_indices.append(labels)
            pred_class_names.extend(pred_label_names)
            true_class_names.extend(true_label_names)
            sentences.extend(batch_sentences)
            predicted_labels_indices.extend(top_indices_list)
            all_pred_probs.append(normalized_probs)

        # contains predicted probs for all the instances
        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        true_labels_indices = torch.cat(true_labels_indices, dim=0).squeeze()

        output_analytics[
            "true_labels_indices"
        ] = true_labels_indices  # torch.LongTensor N, 1
        output_analytics["predicted_labels_indices"] = predicted_labels_indices
        output_analytics["pred_class_names"] = pred_class_names
        output_analytics["true_class_names"] = true_class_names
        output_analytics["sentences"] = sentences
        output_analytics["all_pred_probs"] = all_pred_probs

        return output_analytics

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

    def print_prf_table(self):
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

    def infer_batch(self, lines: Union[List[str], str]):
        if isinstance(lines, str):
            lines = [lines]

        word_tokenizer = WordTokenizer()
        word_vocab = Vocab.load_from_file(self.vocab_store_location)
        max_word_length = self.max_length
        word_add_start_end_token = True
        classnames2idx = self.dataset_class.get_classname2idx()
        idx2classnames = {idx: class_ for class_, idx in classnames2idx.items()}

        iter_dict = self.dataset_class.get_iter_dict(
            lines=lines,
            word_vocab=word_vocab,
            word_tokenizer=word_tokenizer,
            max_word_length=max_word_length,
            word_add_start_end_token=word_add_start_end_token,
        )

        if len(lines) == 1:
            iter_dict["tokens"] = iter_dict["tokens"].unsqueeze(0)

        model_forward_dict = self.model(
            iter_dict, is_training=False, is_validation=False, is_test=True
        )

        # 1 * C - Number of classes
        normalized_probs = model_forward_dict["normalized_probs"]
        top_probs, top_indices = torch.topk(normalized_probs, k=1, dim=1)
        top_indices = top_indices.squeeze().tolist()
        top_indices = [top_indices] if isinstance(top_indices, int) else top_indices
        classnames = [idx2classnames[top_index] for top_index in top_indices]

        return classnames

    def on_user_input(self, line: str) -> str:
        return self.infer_batch(lines=line)[0]
