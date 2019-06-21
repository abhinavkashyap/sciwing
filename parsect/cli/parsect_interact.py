import questionary
from questionary import Choice
from typing import List
from parsect.infer.bow_random_emb_lc_parsect_infer import (
    get_random_emb_linear_classifier_infer_parsect,
)
from parsect.infer.bow_glove_emb_lc_parsect_infer import get_glove_emb_lc_parsect_infer
from parsect.infer.bow_elmo_emb_lc_parsect_infer import get_elmo_emb_lc_infer_parsect
from parsect.infer.bow_elmo_emb_lc_gensect_infer import get_elmo_emb_lc_infer_gensect
from parsect.infer.bow_bert_emb_lc_parsect_infer import (
    get_bow_bert_emb_lc_parsect_infer,
)
from parsect.infer.bow_bert_emb_lc_gensect_infer import (
    get_bow_bert_emb_lc_gensect_infer,
)
from parsect.infer.bi_lstm_lc_infer_gensect import get_bilstm_lc_infer_gensect
from parsect.infer.bi_lstm_lc_infer_parsect import get_bilstm_lc_infer_parsect
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
import pathlib
import pandas as pd

PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]


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
            "elmo-embedding-bow-encoder-linear-classifier-parsect",
            "elmo-embedding-bow-encoder-linear-classifier-gensect",
            "bert-embedding-bow-encoder-linear-classifier-parsect",
            "bert-embedding-bow-encoder-linear-classifier-gensect",
            "bi-lstm-random-emb-linear-classifier-parsect",
            "bi-lstm-random-emb-linear-classifier-gensect",
            "elmo-bilstm-linear-classifier",
            "bert-seq-classifier",
        ]
        self.model_type2exp_prefix = {
            "parsect-random-embedding-bow-encoder-linear-classifier": "parsect_bow_random_emb_lc",
            "genericsect-random-embedding-bow-encoder-linear-classifier": "gensect_bow_random_emb_lc",
            "parsect-glove-embedding-bow-encoder-linear-classifier": "parsect_bow_glove_emb_lc",
            "genericsect-glove-embedding-bow-encoder-linear-classifier": "gensect_bow_glove_emb_lc",
            "elmo-embedding-bow-encoder-linear-classifier-parsect": "parsect_bow_elmo_emb_lc",
            "elmo-embedding-bow-encoder-linear-classifier-gensect": "gensect_bow_elmo_emb_lc",
            "bert-embedding-bow-encoder-linear-classifier-parsect": "parsect_bow_bert",
            "bert-embedding-bow-encoder-linear-classifier-gensect": "gensect_bow_bert",
            "bi-lstm-random-emb-linear-classifier-parsect": "parsect_bi_lstm_lc",
            "bi-lstm-random-emb-linear-classifier-gensect": "gensect_bi_lstm_lc",
            "elmo-bilstm-linear-classifier": "parsect_elmo_bi_lstm_lc",
            "bert-seq-classifier": "parsect_bert_seq",
        }
        self.model_type2inf_func = {
            "parsect-random-embedding-bow-encoder-linear-classifier": get_random_emb_linear_classifier_infer_parsect,
            "genericsect-random-embedding-bow-encoder-linear-classifier": get_random_emb_linear_classifier_infer_genericsect,
            "parsect-glove-embedding-bow-encoder-linear-classifier": get_glove_emb_lc_parsect_infer,
            "genericsect-glove-embedding-bow-encoder-linear-classifier": get_glove_emb_lc_genericsect_infer,
            "elmo-embedding-bow-encoder-linear-classifier-parsect": get_elmo_emb_lc_infer_parsect,
            "elmo-embedding-bow-encoder-linear-classifier-gensect": get_elmo_emb_lc_infer_gensect,
            "bert-embedding-bow-encoder-linear-classifier-parsect": get_bow_bert_emb_lc_parsect_infer,
            "bert-embedding-bow-encoder-linear-classifier-gensect": get_bow_bert_emb_lc_gensect_infer,
            "bi-lstm-random-emb-linear-classifier-parsect": get_bilstm_lc_infer_parsect,
            "bi-lstm-random-emb-linear-classifier-gensect": get_bilstm_lc_infer_gensect,
            "elmo-bilstm-linear-classifier": get_elmo_bilstm_lc_infer,
            "bert-seq-classifier": get_bert_seq_classifier_infer,
        }
        self.s3util = S3Util(os.path.join(AWS_CRED_DIR, "aws_s3_credentials.json"))
        self.msg_printer = wasabi.Printer()
        self.model_type_answer = self.ask_model_type()
        self.generate_report_or_interact = self.ask_generate_report_or_interact()
        if self.generate_report_or_interact == "interact":
            self.interact()
        elif self.generate_report_or_interact == "gen-report":
            self.generate_report()

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

    @staticmethod
    def ask_generate_report_or_interact():
        generate_report_or_interact = questionary.rawselect(
            "What would you like to do ",
            qmark="❓",
            choices=[
                Choice("Interact with model", "interact"),
                Choice("Generate report (for all experiments)", "gen-report"),
            ],
        )
        return generate_report_or_interact.ask()

    def interact(self):
        exp_dir = self.get_experiment_choice()
        inference_func = self.model_type2inf_func[self.model_type_answer]
        inference_client = inference_func(exp_dir)

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
                inference_client.print_confusion_matrix()
            elif interaction_choice == "See-examples-of-Classifications":
                misclassification_choice = questionary.text(
                    "Enter Two Classes separated by a space. [Hint: 1 2]"
                ).ask()
                two_classes = [
                    int(class_) for class_ in misclassification_choice.split()
                ]
                first_class, second_class = two_classes[0], two_classes[1]
                sentences = inference_client.get_misclassified_sentences(
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
                inference_client.print_prf_table()
            elif interaction_choice == "exit":
                self.msg_printer.text("See you again!")
                exit(0)

    def get_experiment_choice(self):
        output_dirpath = pathlib.Path(OUTPUT_DIR)
        experiment_dirnames = self.get_experiments_folder_names()

        if len(experiment_dirnames) == 0:
            self.msg_printer.fail(
                f"There are no experiments for the model type {self.model_type_answer}"
            )
            exit(1)

        experiment_choices = [Choice(dirname) for dirname in experiment_dirnames]
        exp_choice = questionary.rawselect(
            "Please select an experiment", choices=experiment_choices, qmark="❓"
        ).ask()

        exp_choice_path = pathlib.Path(OUTPUT_DIR, exp_choice)
        if not exp_choice_path.is_dir():
            with self.msg_printer.loading(
                f"Downloading experiment {exp_choice} from s3"
            ):
                self.s3util.download_folder(exp_choice)
        return str(output_dirpath.joinpath(exp_choice_path))

    def get_experiments_folder_names(self) -> List[str]:
        """
        Returns the experiment folder names from local output folder
        and amazon s3 folder
        Returns just the plain experiment folder names (not the actual path)
        :return:
        """
        output_dirpath = pathlib.Path(OUTPUT_DIR)
        experiment_dirnames = []
        for foldername in output_dirpath.iterdir():
            if re.match(
                self.model_type2exp_prefix[self.model_type_answer], str(foldername.name)
            ):
                experiment_dirnames.append(foldername.name)

        s3_folder_names = self.s3util.search_folders_with(
            self.model_type2exp_prefix[self.model_type_answer]
        )
        for folder_name in s3_folder_names:
            experiment_dirnames.append(folder_name)

        return experiment_dirnames

    def generate_report(self):
        experiment_dirnames = self.get_experiments_folder_names()
        all_fscores = {}
        row_names = None
        for exp_dirname in experiment_dirnames:
            folder_path = pathlib.Path(OUTPUT_DIR, exp_dirname)
            inference_func = self.model_type2inf_func[self.model_type_answer]
            inference_client = inference_func(str(folder_path))
            fscores, row_names = inference_client.generate_report_for_paper()
            all_fscores[exp_dirname] = fscores

        fscores_df = pd.DataFrame(all_fscores)
        fscores_df.index = row_names
        output_filename = pathlib.Path(
            REPORTS_DIR,
            self.model_type2exp_prefix[self.model_type_answer] + "_report.csv",
        )
        fscores_df.to_csv(output_filename, index=True, header=list(fscores_df.columns))


if __name__ == "__main__":
    cli = ParsectCli()
