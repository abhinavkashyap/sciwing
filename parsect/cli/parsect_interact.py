import questionary
from questionary import Choice
from typing import List
from parsect.clients.random_emb_bow_linear_classifier_infer import (
    get_random_emb_linear_classifier_infer,
)
import wasabi
import parsect.constants as constants
import os

PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]


class ParsectCli:
    """
    This cli helps in interacting with different models of parsect
    """

    def __init__(self):
        self.trained_model_types = ["random embedding-bow encoder-linear classifier"]
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
        if self.model_type_answer == "random embedding-bow encoder-linear classifier":
            choices = []
            for expname in os.listdir(OUTPUT_DIR):
                if expname.startswith("bow_random_emb_lc"):
                    choices.append(Choice(expname))

            exp_choice = questionary.rawselect(
                "Please select an experiment", choices=choices, qmark="❓"
            ).ask()
            exp_choice = os.path.join(OUTPUT_DIR, exp_choice)
            inference = get_random_emb_linear_classifier_infer(exp_choice)
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
