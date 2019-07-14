from typing import Dict, Any, List
import torch.nn as nn
from parsect.infer.BaseInference import BaseInference
from parsect.metrics.token_cls_accuracy import TokenClassificationAccuracy
from torch.utils.data import DataLoader
from parsect.utils.tensor import move_to_device
import torch
import pandas as pd


class ParscitInference(BaseInference):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        hyperparam_config_filepath: str,
        dataset,
    ):
        super(ParscitInference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            hyperparam_config_filepath=hyperparam_config_filepath,
            dataset=dataset,
        )
        self.labelname2idx_mapping = self.test_dataset.get_classname2idx()
        self.idx2labelname_mapping = {
            idx: label_name for label_name, idx in self.labelname2idx_mapping.items()
        }
        ignore_indices = [
            self.labelname2idx_mapping["starting"],
            self.labelname2idx_mapping["ending"],
            self.labelname2idx_mapping["padding"],
        ]
        self.metrics_calculator = TokenClassificationAccuracy(
            idx2labelname_mapping=self.idx2labelname_mapping,
            mask_label_indices=ignore_indices,
        )
        self.load_model()

        with self.msg_printer.loading("Running inference on test data"):
            self.output_analytics = self.run_inference()
        self.msg_printer.good("Finished running inference on test data")

        self.output_df = pd.DataFrame(
            {
                "true_tag_indices": self.output_analytics["true_tag_indices"],
                "predicted_tag_indices": self.output_analytics["predicted_tag_indices"],
                "true_tag_names": self.output_analytics["true_tag_names"],
                "predicted_tag_names": self.output_analytics["predicted_tag_names"],
                "sentences": self.output_analytics["sentences"],
            }
        )

    def run_inference(self) -> Dict[str, Any]:
        loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        output_analytics = {}
        sentences = []  # all the sentences that is seen till now
        predicted_tag_indices = []
        predicted_tag_names = []  # all the tags that are predicted for the sentences
        true_tag_indices = []
        true_tag_names = []

        for iter_dict in loader:
            iter_dict = move_to_device(iter_dict, cuda_device=self.device)

            with torch.no_grad():
                model_output_dict = self.model(
                    iter_dict, is_training=False, is_validation=False, is_test=True
                )

            self.metrics_calculator.calc_metric(
                iter_dict=iter_dict, model_forward_dict=model_output_dict
            )
            tokens = iter_dict["tokens"]
            labels = iter_dict["label"]
            tokens_list = tokens.tolist()
            labels_list = labels.tolist()

            batch_sentences = list(
                map(self.test_dataset.get_disp_sentence_from_indices, tokens_list)
            )
            sentences.extend(batch_sentences)

            predicted_tags = model_output_dict["predicted_tags"]  # List[List[str]]
            predicted_tag_strings = map(
                lambda tags: " ".join(
                    self.test_dataset.get_class_names_from_indices(tags)
                ),
                predicted_tags,
            )
            predicted_tag_strings = list(predicted_tag_strings)

            true_labels_strings = map(
                lambda tags: " ".join(
                    self.test_dataset.get_class_names_from_indices(tags)
                ),
                labels_list,
            )
            true_labels_strings = list(true_labels_strings)

            predicted_tag_indices.extend(predicted_tags)
            predicted_tag_names.extend(predicted_tag_strings)
            true_tag_indices.extend(labels_list)
            true_tag_names.extend(true_labels_strings)

        output_analytics["true_tag_indices"] = true_tag_indices
        output_analytics["predicted_tag_indices"] = predicted_tag_indices
        output_analytics["true_tag_names"] = true_tag_names
        output_analytics["predicted_tag_names"] = predicted_tag_names
        output_analytics["sentences"] = sentences
        return output_analytics

    def print_confusion_matrix(self) -> None:
        self.metrics_calculator.print_confusion_metrics(
            true_tag_indices=self.output_df["true_tag_indices"].tolist(),
            predicted_tag_indices=self.output_df["predicted_tag_indices"].tolist(),
        )

    def print_prf_table(self) -> None:
        prf_table = self.metrics_calculator.report_metrics()
        print(prf_table)

    def get_misclassified_sentences(
        self, first_class: int, second_class: int
    ) -> List[str]:
        return ["", ""]

    def generate_report_for_paper(self):
        paper_report, row_names = self.metrics_calculator.report_metrics(
            report_type="paper"
        )
        return paper_report, row_names
