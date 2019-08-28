import questionary
from questionary import Choice
from typing import List
from sciwing.infer.bow_lc_parsect_infer import get_bow_lc_parsect_infer

from sciwing.infer.bow_elmo_emb_lc_parsect_infer import get_elmo_emb_lc_infer_parsect
from sciwing.infer.bow_elmo_emb_lc_gensect_infer import get_elmo_emb_lc_infer_gensect
from sciwing.infer.bow_bert_emb_lc_parsect_infer import (
    get_bow_bert_emb_lc_parsect_infer,
)
from sciwing.infer.bow_bert_emb_lc_gensect_infer import (
    get_bow_bert_emb_lc_gensect_infer,
)
from sciwing.infer.bi_lstm_lc_infer_gensect import get_bilstm_lc_infer_gensect
from sciwing.infer.bi_lstm_lc_infer_parsect import get_bilstm_lc_infer_parsect
from sciwing.infer.elmo_bi_lstm_lc_infer import get_elmo_bilstm_lc_infer
from sciwing.infer.bow_lc_gensect_infer import get_bow_lc_gensect_infer
from sciwing.infer.bilstm_crf_infer import get_bilstm_crf_infer
from sciwing.infer.science_ie_infer import get_science_ie_infer
import wasabi
import sciwing.constants as constants
from sciwing.utils.amazon_s3 import S3Util
from sciwing.utils.science_ie_eval import calculateMeasures
import os
import re
import pathlib
import pandas as pd

PATHS = constants.PATHS
FILES = constants.FILES
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class ParsectCli:
    """
    This cli helps in interacting with different models of sciwing
    """

    def __init__(self):
        self.trained_model_types = [
            "sciwing-random-embedding-bow-encoder-linear-classifier",
            "genericsect-random-embedding-bow-encoder-linear-classifier",
            "sciwing-glove-embedding-bow-encoder-linear-classifier",
            "genericsect-glove-embedding-bow-encoder-linear-classifier",
            "elmo-embedding-bow-encoder-linear-classifier-sciwing",
            "elmo-embedding-bow-encoder-linear-classifier-gensect",
            "bert-embedding-bow-encoder-linear-classifier-sciwing",
            "bert-embedding-bow-encoder-linear-classifier-gensect",
            "bi-lstm-random-emb-linear-classifier-sciwing",
            "bi-lstm-random-emb-linear-classifier-gensect",
            "elmo-bilstm-linear-classifier",
            "lstm-crf-parscit-tagger",
            "lstm-crf-scienceie-tagger",
        ]
        self.model_type2exp_prefix = {
            "sciwing-random-embedding-bow-encoder-linear-classifier": "parsect_bow_random_emb_lc",
            "genericsect-random-embedding-bow-encoder-linear-classifier": "gensect_bow_random_emb_lc",
            "sciwing-glove-embedding-bow-encoder-linear-classifier": "parsect_bow_glove_emb_lc",
            "genericsect-glove-embedding-bow-encoder-linear-classifier": "gensect_bow_glove_emb_lc",
            "elmo-embedding-bow-encoder-linear-classifier-sciwing": "parsect_bow_elmo_emb_lc",
            "elmo-embedding-bow-encoder-linear-classifier-gensect": "gensect_bow_elmo_emb_lc",
            "bert-embedding-bow-encoder-linear-classifier-sciwing": "parsect_bow_bert",
            "bert-embedding-bow-encoder-linear-classifier-gensect": "gensect_bow_bert",
            "bi-lstm-random-emb-linear-classifier-sciwing": "parsect_bi_lstm_lc",
            "bi-lstm-random-emb-linear-classifier-gensect": "gensect_bi_lstm_lc",
            "elmo-bilstm-linear-classifier": "parsect_elmo_bi_lstm_lc",
            "lstm-crf-parscit-tagger": "lstm_crf_parscit",
            "lstm-crf-scienceie-tagger": "lstm_crf_scienceie",
        }
        self.model_type2inf_func = {
            "sciwing-random-embedding-bow-encoder-linear-classifier": get_bow_lc_parsect_infer,
            "genericsect-random-embedding-bow-encoder-linear-classifier": get_bow_lc_gensect_infer,
            "sciwing-glove-embedding-bow-encoder-linear-classifier": get_bow_lc_parsect_infer,
            "genericsect-glove-embedding-bow-encoder-linear-classifier": get_bow_lc_gensect_infer,
            "elmo-embedding-bow-encoder-linear-classifier-sciwing": get_elmo_emb_lc_infer_parsect,
            "elmo-embedding-bow-encoder-linear-classifier-gensect": get_elmo_emb_lc_infer_gensect,
            "bert-embedding-bow-encoder-linear-classifier-sciwing": get_bow_bert_emb_lc_parsect_infer,
            "bert-embedding-bow-encoder-linear-classifier-gensect": get_bow_bert_emb_lc_gensect_infer,
            "bi-lstm-random-emb-linear-classifier-sciwing": get_bilstm_lc_infer_parsect,
            "bi-lstm-random-emb-linear-classifier-gensect": get_bilstm_lc_infer_gensect,
            "elmo-bilstm-linear-classifier": get_elmo_bilstm_lc_infer,
            "lstm-crf-parscit-tagger": get_bilstm_crf_infer,
            "lstm-crf-scienceie-tagger": get_science_ie_infer,
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
        choices = [
            Choice("Interact with model", "interact"),
            Choice("Generate report (for all experiments)", "gen-report"),
        ]

        generate_report_or_interact = questionary.rawselect(
            "What would you like to do ", qmark="❓", choices=choices
        )
        return generate_report_or_interact.ask()

    def interact(self):
        exp_dir = self.get_experiment_choice()
        exp_dir_path = pathlib.Path(exp_dir)
        inference_func = self.model_type2inf_func[self.model_type_answer]
        inference_client = inference_func(exp_dir)
        inference_client.run_test()

        while True:
            choices = [
                Choice("See-Confusion-Matrix"),
                Choice("See-examples-of-Classifications"),
                Choice("See-prf-table"),
                Choice(title="Enter text ", value="enter_text"),
                Choice("exit"),
            ]
            if self.model_type_answer == "lstm-crf-scienceie-tagger":
                choices.append(Choice("official-results", "semeval_official_results"))

            interaction_choice = questionary.rawselect(
                "What would you like to do now", qmark="❓", choices=choices
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
                    f"Sentences with class {first_class} classified as {second_class}".capitalize()
                )

                for sentence in sentences:
                    print(sentence)

                self.msg_printer.divider("")
            elif interaction_choice == "See-prf-table":
                inference_client.print_metrics()

            elif interaction_choice == "enter_text":
                text = questionary.text("Enter Text: ").ask()
                tagged_string = inference_client.on_user_input(text)
                print(tagged_string)

            elif interaction_choice == "semeval_official_results":
                dev_folder = pathlib.Path(SCIENCE_IE_DEV_FOLDER)
                pred_folder = pathlib.Path(
                    REPORTS_DIR, f"science_ie_{exp_dir_path.stem}_results"
                )
                if not pred_folder.is_dir():
                    pred_folder.mkdir()
                inference_client.generate_predict_folder(
                    dev_folder=dev_folder, pred_folder=pred_folder
                )
                calculateMeasures(
                    folder_gold=str(dev_folder),
                    folder_pred=str(pred_folder),
                    remove_anno="rel",
                )
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
                self.s3util.download_folder(
                    exp_choice, download_only_best_checkpoint=True
                )
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

        if len(experiment_dirnames) == 0:
            self.msg_printer.fail(
                f"There are no experiments for the model type {self.model_type_answer}"
            )
            exit(1)

        all_fscores = {}
        row_names = None
        for exp_dirname in experiment_dirnames:
            folder_path = pathlib.Path(OUTPUT_DIR, exp_dirname)
            if not folder_path.is_dir():
                with self.msg_printer.loading(
                    f"Downloading experiment {exp_dirname} from s3"
                ):
                    self.s3util.download_folder(
                        exp_dirname, download_only_best_checkpoint=True
                    )

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
