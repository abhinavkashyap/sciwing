import sciwing.constants as constants
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from sciwing.infer.classification.BaseClassificationInference import (
    BaseClassificationInference,
)
from sciwing.data.datasets_manager import DatasetsManager
from deprecated import deprecated
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import Any, Dict, List
import pandas as pd
from sciwing.data.line import Line
from sciwing.data.label import Label
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
        self,
        model: nn.Module,
        model_filepath: str,
        datasets_manager: DatasetsManager,
        tokens_namespace: str = "tokens",
        normalized_probs_namespace: str = "normalized_probs",
        device: str = "cpu",
    ):

        super(ClassificationInference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            datasets_manager=datasets_manager,
            device=device,
        )
        self.batch_size = 32
        self.tokens_namespace = tokens_namespace
        self.normalized_probs_namespace = normalized_probs_namespace
        self.label_namespace = self.datasets_manager.label_namespaces[0]

        self.labelname2idx_mapping = self.datasets_manager.get_label_idx_mapping(
            label_namespace=self.label_namespace
        )
        self.idx2labelname_mapping = self.datasets_manager.get_idx_label_mapping(
            label_namespace=self.label_namespace
        )

        self.load_model()

        self.metrics_calculator = PrecisionRecallFMeasure(
            datasets_manager=datasets_manager
        )
        self.output_analytics = None

        # create a dataframe with all the information
        self.output_df = None

    def run_inference(self) -> Dict[str, Any]:

        with self.msg_printer.loading(text="Running inference on test data"):
            loader = DataLoader(
                dataset=self.datasets_manager.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=list,
            )
            output_analytics = {}

            # contains the predicted class names for all the instances
            pred_class_names = []
            true_class_names = []  # contains the true class names for all the instances
            sentences = []  # batch sentences in english
            true_labels_indices = []
            predicted_labels_indices = []
            all_pred_probs = []
            self.metrics_calculator.reset()

            for lines_labels in loader:
                lines_labels = list(zip(*lines_labels))
                lines = lines_labels[0]
                labels = lines_labels[1]

                batch_sentences = [line.text for line in lines]
                model_output_dict = self.model_forward_on_lines(lines=lines)
                normalized_probs = model_output_dict[self.normalized_probs_namespace]
                self.metrics_calculator.calc_metric(
                    lines=lines, labels=labels, model_forward_dict=model_output_dict
                )
                true_label_ind, true_label_names = self.get_true_label_indices_names(
                    labels=labels
                )
                (
                    pred_label_indices,
                    pred_label_names,
                ) = self.model_output_dict_to_prediction_indices_names(
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

        self.msg_printer.good(title="Finished running inference")
        return output_analytics

    def model_forward_on_lines(self, lines: List[Line]):
        with torch.no_grad():
            model_output_dict = self.model(
                lines=lines, is_training=False, is_validation=False, is_test=True
            )
        return model_output_dict

    def get_misclassified_sentences(self, true_label_idx: int, pred_label_idx: int):
        """This returns the true label misclassified as
        pred label idx

        Parameters
        ----------
        true_label_idx : int
            The label index of the true class name
        pred_label_idx : int
            The label index of the predicted class name


        Returns
        -------
        List[str]
            A list of strings where the true class is classified as pred class.

        """

        instances_idx = self.output_df[
            self.output_df["true_labels_indices"].isin([true_label_idx])
            & self.output_df["predicted_labels_indices"].isin([pred_label_idx])
        ].index.tolist()

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

            print(stylized_sentence)

    def print_confusion_matrix(self) -> None:
        """ Prints the confusion matrix for the test dataset
        """
        self.metrics_calculator.print_confusion_metrics(
            predicted_probs=self.output_analytics["all_pred_probs"],
            labels=self.output_analytics["true_labels_indices"].unsqueeze(1),
        )

    def report_metrics(self):
        metrics = self.metrics_calculator.report_metrics()
        for namespace, table in metrics.items():
            self.msg_printer.divider(f"Results for {namespace.upper()}")
            print(table)

    @deprecated(reason="This method is deprecated. It will be removed in version 0.1")
    def generate_report_for_paper(self):
        """ Generates just the fscore to be used in reporting on print

        """
        paper_report = self.metrics_calculator.report_metrics(report_type="paper")
        class_numbers = sorted(self.idx2labelname_mapping.keys(), reverse=False)
        row_names = [
            f"class_{class_num} - ({self.idx2labelname_mapping[class_num]})"
            for class_num in class_numbers
        ]
        row_names.extend([f"Micro-Fscore", f"Macro-Fscore"])
        return paper_report, row_names

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

    def infer_batch(self, lines: List[str]) -> List[str]:
        """ Runs inference on a batch of lines
        This method can be used for applications. When APIS are being developed
        to serve over the web or when terminal applications are being written
        to read from files and infer, this method comes in handy


        Parameters
        ----------
        lines : List[str]
            List of text spans to be infered

        Returns
        -------
        List[str]
            Reutrns the class names for all the sentences in the input

        """
        lines = [self.datasets_manager.make_line(line=line) for line in lines]

        model_output_dict = self.model_forward_on_lines(lines=lines)
        _, pred_classnames = self.model_output_dict_to_prediction_indices_names(
            model_output_dict=model_output_dict
        )
        return pred_classnames

    def on_user_input(self, line: str) -> str:
        """ Runs the inference when the user inputs a single sentence either on the terminal
        or some other application

        Parameters
        ----------
        line : str
            The line entered by the user

        Returns
        -------
        str
            The class label that is infered for the user input

        """
        return self.infer_batch(lines=[line])[0]

    def get_true_label_indices_names(
        self, labels: List[Label]
    ) -> (List[int], List[str]):
        label_names = [label.text for label in labels]
        label_indices = [
            self.labelname2idx_mapping[label_name] for label_name in label_names
        ]
        return label_indices, label_names

    def run_test(self):
        """ Runs inference and reports test metrics
        """
        self.output_analytics = self.run_inference()
        self.output_df = pd.DataFrame(self.output_analytics)
