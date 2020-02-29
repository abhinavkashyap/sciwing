import torch.nn as nn
from typing import Optional, Union, Dict, Any, List
import torch
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
from sciwing.infer.seq_label_inference.BaseSeqLabelInference import (
    BaseSeqLabelInference,
)
from sciwing.utils.science_ie_data_utils import ScienceIEDataUtils
import wasabi
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from sciwing.utils.vis_seq_tags import VisTagging
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
import pathlib


class SequenceLabellingInference(BaseSeqLabelInference):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        datasets_manager: DatasetsManager,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        predicted_tags_namespace_prefix: str = "predicted_tags",
    ):
        super(SequenceLabellingInference, self).__init__(
            model=model,
            model_filepath=model_filepath,
            datasets_manager=datasets_manager,
            device=device,
        )

        self.predicted_tags_namespace_prefix = predicted_tags_namespace_prefix
        self.labels_namespaces = self.datasets_manager.label_namespaces
        self.msg_printer = wasabi.Printer()
        self.metrics_calculator = TokenClassificationAccuracy(
            datasets_manager=datasets_manager
        )

        # The key is the namespace of different labels
        # The value is a dictioary of label->idx
        self.label2idx_mapping: Dict[str, Dict[str, Any]] = {}
        self.idx2label_mapping: Dict[str, Dict[str, Any]] = {}
        for namespace in self.labels_namespaces:
            self.label2idx_mapping[
                namespace
            ] = self.datasets_manager.get_label_idx_mapping(label_namespace=namespace)
            self.idx2label_mapping[
                namespace
            ] = self.datasets_manager.get_idx_label_mapping(label_namespace=namespace)

        self.output_analytics = None
        self.output_df = None
        self.batch_size = 32
        self.load_model()

        self.namespace_to_unique_categories = {}
        self.namespace_to_visualizer = {}
        for namespace in self.labels_namespaces:
            categories = list(
                set([label for label in self.label2idx_mapping[namespace].keys()])
            )
            visualizer = VisTagging(tags=categories)
            self.namespace_to_unique_categories[namespace] = categories
            self.namespace_to_visualizer[namespace] = visualizer

    def run_inference(self):
        with self.msg_printer.loading(text="Running inference on test data"):
            loader = DataLoader(
                dataset=self.datasets_manager.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=list,
            )

            # output analytics for every label namespace
            output_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
            sentences = []  # all the sentences that is seen till now
            predicted_tag_indices: Dict[str, list] = defaultdict(list)

            # namespace -> all the tags that are predicted for the sentences
            predicted_tag_names: Dict[str, list] = defaultdict(list)
            true_tag_indices: Dict[str, list] = defaultdict(list)
            true_tag_names: Dict[str, list] = defaultdict(list)
            self.metrics_calculator.reset()

            for lines_labels in loader:
                lines_labels_ = list(zip(*lines_labels))
                lines = lines_labels_[0]
                labels = lines_labels_[1]

                batch_sentences = [line.text for line in lines]
                model_output_dict = self.model_forward_on_lines(lines=lines)
                self.metrics_calculator.calc_metric(
                    lines=lines, labels=labels, model_forward_dict=model_output_dict
                )
                sentences.extend(batch_sentences)

                (
                    predicted_tags,
                    predicted_tags_strings,
                ) = self.model_output_dict_to_prediction_indices_names(
                    model_output_dict=model_output_dict
                )

                true_tags, true_labels_strings = self.get_true_label_indices_names(
                    labels=labels
                )

                for namespace in self.labels_namespaces:
                    predicted_tag_indices[namespace].extend(predicted_tags[namespace])
                    predicted_tag_names[namespace].extend(
                        predicted_tags_strings[namespace]
                    )
                    true_tag_indices[namespace].extend(true_tags[namespace])
                    true_tag_names[namespace].extend(true_labels_strings[namespace])

            for namespace in self.labels_namespaces:
                output_analytics[namespace]["true_tag_indices"] = true_tag_indices[
                    namespace
                ]
                output_analytics[namespace][
                    "predicted_tag_indices"
                ] = predicted_tag_indices[namespace]
                output_analytics[namespace]["true_tag_names"] = true_tag_names[
                    namespace
                ]
                output_analytics[namespace][
                    "predicted_tag_names"
                ] = predicted_tag_names[namespace]
                output_analytics[namespace]["sentences"] = sentences

            return output_analytics

    def model_forward_on_lines(self, lines: List[Line]):
        with torch.no_grad():
            model_output_dict = self.model(
                lines=lines,
                labels=None,
                is_training=False,
                is_validation=False,
                is_test=True,
            )
        return model_output_dict

    def model_output_dict_to_prediction_indices_names(
        self, model_output_dict: Dict[str, Any]
    ) -> (Dict[str, List[int]], Dict[str, List[str]]):
        predicted_tags_indices_ = defaultdict(list)
        predicted_tags_strings_ = defaultdict(list)
        for namespace in self.labels_namespaces:
            # List[List[str]]
            batch_tags = model_output_dict[
                f"{self.predicted_tags_namespace_prefix}_{namespace}"
            ]
            predicted_tags_indices_[namespace].extend(batch_tags)
            for predicted_tags in batch_tags:
                predicted_tag_string = [
                    self.idx2label_mapping[namespace][predicted_tag_idx]
                    for predicted_tag_idx in predicted_tags
                ]
                predicted_tag_string = " ".join(predicted_tag_string)
                predicted_tags_strings_[namespace].append(predicted_tag_string)

        return predicted_tags_indices_, predicted_tags_strings_

    def get_true_label_indices_names(
        self, labels: List[SeqLabel]
    ) -> (Dict[str, List[int]], Dict[str, List[str]]):
        true_labels_indices = defaultdict(list)
        true_labels_names = defaultdict(list)

        for namespace in self.labels_namespaces:
            for label in labels:
                label_ = label.tokens[namespace]
                label_ = [tok.text for tok in label_]
                true_labels_names[namespace].append(" ".join(label_))
                label_indices = [
                    self.label2idx_mapping[namespace][tok] for tok in label_
                ]
                true_labels_indices[namespace].append(label_indices)

        return true_labels_indices, true_labels_names

    def report_metrics(self):
        prf_tables = self.metrics_calculator.report_metrics()
        for namespace in self.labels_namespaces:
            self.msg_printer.divider(f"Report for {namespace}")
            print(prf_tables[namespace])

    def run_test(self):
        self.output_analytics = self.run_inference()
        self.output_df = pd.DataFrame(self.output_analytics)

    def print_confusion_matrix(self):
        """ This prints the confusion metrics for the entire dataset
        Returns
        -------
        None
        """
        for namespace in self.labels_namespaces:
            # List[List[int]]
            true_tags_indices = self.output_analytics[namespace]["true_tag_indices"]
            predicted_tag_indices = self.output_analytics[namespace][
                "predicted_tag_indices"
            ]

            max_len_pred = max([len(pred_tags) for pred_tags in predicted_tag_indices])
            max_len_true = max([len(true_tags) for true_tags in true_tags_indices])

            # pad everything to the max len of both
            max_len = max_len_pred if max_len_pred > max_len_true else max_len_true

            numericalizer = self.datasets_manager.namespace_to_numericalizer[namespace]
            padded_true_tag_indices = numericalizer.pad_batch_instances(
                instances=true_tags_indices,
                max_length=max_len,
                add_start_end_token=False,
            )

            padded_predicted_tag_indices = numericalizer.pad_batch_instances(
                instances=predicted_tag_indices,
                max_length=max_len,
                add_start_end_token=False,
            )

            labels_mask = numericalizer.get_mask_for_batch_instances(
                instances=padded_true_tag_indices
            )
            # we have to pad the true tags indices and predicted tag indices all to max length

            self.metrics_calculator.print_confusion_metrics(
                true_tag_indices=padded_true_tag_indices,
                predicted_tag_indices=padded_predicted_tag_indices,
                labels_mask=labels_mask,
            )

    def get_misclassified_sentences(self, true_label_idx: int, pred_label_idx: int):

        for namespace in self.labels_namespaces:
            self.msg_printer.divider(f"Namespace {namespace.lower()}")

            true_tag_indices = self.output_df[namespace].true_tag_indices
            pred_tag_indices = self.output_df[namespace].predicted_tag_indices

            indices = []

            for idx, (true_tag_index, pred_tag_index) in enumerate(
                zip(true_tag_indices, pred_tag_indices)
            ):
                true_tags_pred_tags = zip(true_tag_index, pred_tag_index)
                for true_tag, pred_tag in true_tags_pred_tags:
                    if true_tag == true_label_idx and pred_tag == pred_label_idx:
                        indices.append(idx)
                        break

            for idx in indices:
                sentence = self.output_analytics[namespace]["sentences"][idx].split()
                true_labels = self.output_analytics[namespace]["true_tag_names"][
                    idx
                ].split()
                pred_labels = self.output_analytics[namespace]["predicted_tag_names"][
                    idx
                ].split()
                len_sentence = len(sentence)
                true_labels = true_labels[:len_sentence]
                pred_labels = pred_labels[:len_sentence]
                stylized_string_true = self.namespace_to_visualizer[
                    namespace
                ].visualize_tokens(sentence, true_labels)
                stylized_string_predicted = self.namespace_to_visualizer[
                    namespace
                ].visualize_tokens(sentence, pred_labels)

                sentence = (
                    f"GOLD LABELS \n{'*' * 80} \n{stylized_string_true} \n\n"
                    f"PREDICTED LABELS \n{'*' * 80} \n{stylized_string_predicted}\n\n"
                )
                print(sentence)

    def on_user_input(self, line: Union[Line, str]) -> Dict[str, List[str]]:
        return self.infer_batch(lines=[line])

    def infer_batch(self, lines: Union[List[Line], List[str]]) -> Dict[str, List[str]]:
        lines_ = []

        if isinstance(lines[0], str):
            for line in lines:
                line_ = self.datasets_manager.make_line(line=line)
                lines_.append(line_)

        else:
            lines_ = lines

        model_output_dict = self.model_forward_on_lines(lines=lines_)
        _, pred_classnames = self.model_output_dict_to_prediction_indices_names(
            model_output_dict
        )
        return pred_classnames

    def generate_scienceie_prediction_folder(
        self, dev_folder: pathlib.Path, pred_folder: pathlib.Path
    ):
        """ Generates the predicted folder for the dataset in the test folder
        for ScienceIE. This is very specific to ScienceIE. Not meant to use
        with other tasks

        ScienceIE is a SemEval Task that needs the files to be written into a
        folder and it reports metrics by reading files from that folder. This
        method generates the predicted folder given the dev folder


        Parameters
        ----------
        dev_folder : pathlib.Path
            The path where the dev files are present
        pred_folder : pathlib.Path
            The path where the predicted files will be written

        Returns
        -------

        """
        science_ie_data_utils = ScienceIEDataUtils(
            folderpath=dev_folder, ignore_warnings=True
        )
        file_ids = science_ie_data_utils.get_file_ids()

        for file_id in file_ids:
            with self.msg_printer.loading(
                f"Generating Science IE results for file {file_id}"
            ):
                text = science_ie_data_utils.get_text_from_fileid(file_id)
                sents = science_ie_data_utils.get_sents(text)
                try:
                    assert bool(text.split()), f"File {file_id} does not have any text"
                except AssertionError:
                    continue

                try:
                    assert len(sents) > 0
                except AssertionError:
                    continue

                conll_filepath = pred_folder.joinpath(f"{file_id}.conll")
                ann_filepath = pred_folder.joinpath(f"{file_id}.ann")
                conll_lines = []

                for sent in sents:
                    line = [token.text for token in sent]
                    line = " ".join(line)
                    prediction_classnames = self.on_user_input(line=line)

                    tag_names = [line.split()]
                    for namespace in ["TASK", "PROCESS", "MATERIAL"]:
                        # List[str] - List of predicted classnames
                        classnames_ = prediction_classnames[namespace][0].split()
                        tag_names.append(classnames_)
                        assert len(line.split()) == len(
                            classnames_
                        ), f"len sent: {len(line.split())}, len task_tag_name: {len(classnames_)}"

                    zipped_text_tag_names = list(zip(*tag_names))

                    for text_tag_name in zipped_text_tag_names:
                        token, task_tag, process_tag, material_tag = text_tag_name
                        conll_line = " ".join(
                            [token, task_tag, process_tag, material_tag]
                        )
                        conll_lines.append(conll_line)

                with open(conll_filepath, "w") as fp:
                    fp.writelines("\n".join(conll_lines))
                    fp.write("\n")

                science_ie_data_utils.write_ann_file_from_conll_file(
                    conll_filepath=conll_filepath, ann_filepath=ann_filepath, text=text
                )
