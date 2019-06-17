import questionary
from questionary import Choice
from typing import List
from parsect.infer.bow_random_emb_lc_parsect_infer import (
    get_random_emb_linear_classifier_infer_parsect,
)
from parsect.infer.bow_glove_emb_lc_parsect_infer import get_glove_emb_lc_parsect_infer
from parsect.infer.elmo_emb_bow_linear_classifier_infer import (
    get_elmo_emb_linear_classifier_infer,
)
from parsect.infer.bert_emb_bow_linear_classifier_infer import (
    get_bert_emb_bow_linear_classifier_infer,
)
from parsect.infer.bi_lstm_lc_infer import get_bilstm_lc_infer
from parsect.infer.elmo_bi_lstm_lc_infer import get_elmo_bilstm_lc_infer
from parsect.infer.bert_seq_classifier_infer import get_bert_seq_classifier_infer
from parsect.infer.bow_random_emb_lc_genericsect_infer import (
    get_random_emb_linear_classifier_infer_genericsect,
)
from parsect.infer.bow_glove_emb_lc_genericsect_infer import (
    get_glove_emb_lc_genericsect_infer,
)
import wasabi
import parsect.constants as constants
from parsect.utils.amazon_s3 import S3Util
import os
import re

PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]


class ParsectCli:
    """
    This cli helps in interacting with different models of parsect
    """

    def __init__(self):
        self.trained_model_types = [
            "parsect-random-embedding-bow-encoder-linear-classifier",
            "genericsect-random-embedding-bow-encoder-linear-classifier",
            "parsect-glove-embedding-bow-encoder-linear-classifier",
            "genericsect-glove-embedding-bow-encoder-linear-classifier",
            "elmo-embedding-bow-encoder-linear_classifier",
            "bert-embedding-bow-encoder-linear-classifier",
            "bi-lstm-random-emb-linear-classifier",
            "elmo-bilstm-linear-classifier",
            "bert-seq-classifier",
        ]
        self.s3util = S3Util(os.path.join(AWS_CRED_DIR, "aws_s3_credentials.json"))
        self.msg_printer = wasabi.Printer()
        self.model_type_answer = self.ask_model_type()
        self.inference_client = self.get_inference()
        self.interact()

    def ask_model_type(self):
        choices = self.return_model_type_choices()
        model_type_question = questionary.rawselect(
            "We have the following trained models. Chose one",
            qmark="❓",
            choices=choices,
        )
        return model_type_question.ask()

    def return_model_type_choices(self) -> List[Choice]:
        choices = []
        for model_type in self.trained_model_types:
            choices.append(Choice(model_type))
        return choices

    def get_inference(self):
        inference = None
        if (
            self.model_type_answer
            == "parsect-random-embedding-bow-encoder-linear-classifier"
        ):
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if expname.startswith("bow_random_emb_lc"):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(".*bow_random.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_dir = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_random_emb_linear_classifier_infer_parsect(exp_dir)

        if (
            self.model_type_answer
            == "genericsect-random-embedding-bow-encoder-linear-classifier"
        ):
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if re.search(".*bow_random_generic_sect.*", expname):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(
                ".*bow_random_generic_sect.*"
            )
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_dir = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_random_emb_linear_classifier_infer_genericsect(exp_dir)

        if (
            self.model_type_answer
            == "parsect-glove-embedding-bow-encoder-linear-classifier"
        ):
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if expname.startswith("bow_glove_emb_lc"):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(".*bow_glove_emb.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_glove_emb_lc_parsect_infer(exp_choice)

        if (
            self.model_type_answer
            == "genericsect-glove-embedding-bow-encoder-linear-classifier"
        ):
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if re.search(".*bow_random_generic_sect.*", expname):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(
                ".*bow_random_generic_sect_.*"
            )
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_glove_emb_lc_genericsect_infer(exp_choice)

        if self.model_type_answer == "elmo-embedding-bow-encoder-linear_classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if bool(re.search(".*bow_elmo_emb_lc_.*", expname)):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(".*bow_elmo_emb.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_elmo_emb_linear_classifier_infer(exp_choice)

        if self.model_type_answer == "bert-embedding-bow-encoder-linear-classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if bool(re.search(".*bow_bert_.*", expname)):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with(".*bow_bert_.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_bert_emb_bow_linear_classifier_infer(exp_choice)

        if self.model_type_answer == "bi-lstm-random-emb-linear-classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if bool(re.match("bi_lstm_lc.*", expname)):
                    choices.append(Choice(expname))
            # search in s3
            folder_names = self.s3util.search_folders_with("bi_lstm_lc.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_bilstm_lc_infer(exp_choice)

        if self.model_type_answer == "elmo-bilstm-linear-classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if bool(re.search(".*elmo_bi_lstm_lc.*", expname)):
                    choices.append(Choice(expname))

            # search in s3
            folder_names = self.s3util.search_folders_with("elmo_bi_lstm_lc.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_elmo_bilstm_lc_infer(exp_choice)

        if self.model_type_answer == "bert-seq-classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if bool(re.search(".*bert_seq_classifier.*", expname)):
                    choices.append(Choice(expname))
            # search in s3
            folder_names = self.s3util.search_folders_with(".*bert_seq_classifier.*")
            for folder_name in folder_names:
                choices.append(Choice(folder_name))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()

            if not os.path.isdir(os.path.join(OUTPUT_DIR, exp_choice)):
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_choice} from s3"
                ):
                    self.s3util.download_folder(exp_choice)

            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_bert_seq_classifier_infer(exp_choice)

        return inference

    def interact(self):
        while True:
            interaction_choice = questionary.rawselect(
                "What would you like to do now",
                qmark="❓",
                choices=[
                    Choice("See-Confusion-Matrix"),
                    Choice("See-examples-of-Classifications"),
                    Choice("See-prf-table"),
                    Choice("exit"),
                ],
            ).ask()
            if interaction_choice == "See-Confusion-Matrix":
                self.inference_client.print_confusion_matrix()
            elif interaction_choice == "See-examples-of-Classifications":
                misclassification_choice = questionary.text(
                    "Enter Two Classes separated by a space. [Hint: 1 2]"
                ).ask()
                two_classes = [
                    int(class_) for class_ in misclassification_choice.split()
                ]
                first_class, second_class = two_classes[0], two_classes[1]
                sentences = self.inference_client.get_misclassified_sentences(
                    first_class, second_class
                )
                self.msg_printer.divider(
                    "Sentences with class {0} misclassified as {1}".format(
                        first_class, second_class
                    )
                )
                if first_class != second_class:
                    for sentence in sentences:
                        self.msg_printer.fail(sentence)
                else:
                    for sentence in sentences:
                        self.msg_printer.good(sentence)
                self.msg_printer.divider("")
            elif interaction_choice == "See-prf-table":
                self.inference_client.print_prf_table()
            elif interaction_choice == "exit":
                self.msg_printer.text("See you again!")
                exit(0)


if __name__ == "__main__":
    cli = ParsectCli()
