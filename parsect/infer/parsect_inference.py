import parsect.constants as constants
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from parsect.infer.BaseInference import BaseInference
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import Any, Dict, List
import pandas as pd
from parsect.utils.tensor import move_to_device

FILES = constants.FILES

SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ParsectInference(BaseInference):
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
    """

    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        hyperparam_config_filepath: str,
        dataset,
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
        super(BaseInference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            hyperparam_config_filepath=hyperparam_config_filepath,
            dataset=dataset,
        )

        self.labelname2idx_mapping = self.test_dataset.get_classname2idx()
        self.idx2labelname_mapping = {
            idx: label_name for label_name, idx in self.labelname2idx_mapping.items()
        }
        self.metrics_calculator = PrecisionRecallFMeasure(
            idx2labelname_mapping=self.idx2labelname_mapping
        )

        self.load_model()

        with self.msg_printer.loading("Running inference on test data"):
            self.output_analytics = self.run_inference()
        self.msg_printer.good("Finished running inference on test data")

        # create a dataframe with all the information
        self.output_df = pd.DataFrame(
            {
                "true_labels_indices": self.output_analytics[
                    "true_labels_indices"
                ].tolist(),
                "pred_class_names": self.output_analytics["pred_class_names"],
                "true_class_names": self.output_analytics["true_class_names"],
                "sentences": self.output_analytics["sentences"],
                "predicted_labels_indices": self.output_analytics[
                    "predicted_labels_indices"
                ],
            }
        )

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
            tokens = iter_dict["tokens"]
            labels = iter_dict["label"]
            labels = labels.squeeze(1)
            labels_list = labels.tolist()
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
        ] = true_labels_indices  # torch.LongTensor
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

        sentences = [self.output_analytics["sentences"][idx] for idx in instances_idx]

        return sentences

    def print_confusion_matrix(self) -> None:
        self.metrics_calculator.print_confusion_metrics(
            predicted_probs=self.output_analytics["all_pred_probs"],
            labels=self.output_analytics["true_labels_indices"],
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