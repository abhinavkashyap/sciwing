from questionary import text as ask_text
from questionary import rawselect
from questionary import Choice
from sciwing.infer.interface_client_base import BaseInterfaceClient
import wasabi
import sciwing.constants as sciwing_constants
from sciwing.utils.science_ie_eval import calculateMeasures
import pathlib


PATHS = sciwing_constants.PATHS
FILES = sciwing_constants.FILES
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
REPORTS_DIR = PATHS["REPORTS_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class SciWINGInteract:
    """
    This cli helps in interacting with different models of sciwing
    """

    def __init__(self, infer_client: BaseInterfaceClient):
        if isinstance(infer_client, BaseInterfaceClient):
            self.infer_obj = infer_client.build_infer()
        else:
            # You can pass the infer obj directly
            # Refer to sciwing.infer.seq_label.BaseSeqLabelInference or sciwing.infer.seq_label.BaseClassificationInference
            self.infer_obj = infer_client
        self.msg_printer = wasabi.Printer()

    def interact(self):
        """ Interact with the user to explore different models

        This method provides various options for exploration of the different models.

        - ``See-Confusion-Matrix`` shows the confusion matrix on the test dataset.
        - ``See-Examples-of-Classification`` is to explore correct and mis-classifications. You can provide two class numbers as in, ``2 3`` and it shows examples in the test dataset where text that belong to class ``2`` is classified as class ``3``.
        - ``See-prf-table`` shows the precision recall and fmeasure per class.
        - ``See-text`` - Manually enter text and look at the classification results.
        """
        self.infer_obj.run_test()

        while True:
            choices = [
                Choice("See-Confusion-Matrix"),
                Choice("See-examples-of-Classifications"),
                Choice("See-prf-table"),
                Choice(title="Enter text ", value="enter_text"),
                Choice(
                    title="If this is ScienceIE chose this to generate results",
                    value="science-ie-official-results",
                ),
                Choice("exit"),
            ]

            interaction_choice = rawselect(
                "What would you like to do now", qmark="❓", choices=choices
            ).ask()

            if interaction_choice == "See-Confusion-Matrix":
                self.infer_obj.print_confusion_matrix()
            elif interaction_choice == "See-examples-of-Classifications":
                misclassification_choice = ask_text(
                    "Enter Two Classes separated by a space. [Hint: 1 2]"
                ).ask()
                two_classes = [
                    int(class_) for class_ in misclassification_choice.split()
                ]
                first_class, second_class = two_classes[0], two_classes[1]
                self.infer_obj.get_misclassified_sentences(first_class, second_class)

            elif interaction_choice == "See-prf-table":
                self.infer_obj.report_metrics()

            elif interaction_choice == "enter_text":
                text = ask_text("Enter Text: ").ask()
                tagged_string = self.infer_obj.on_user_input(text)
                print(tagged_string)

            elif interaction_choice == "semeval_official_results":
                dev_folder = pathlib.Path(SCIENCE_IE_DEV_FOLDER)
                pred_folder = ask_text(
                    message="Enter the directory path for storing results"
                )
                pred_folder = pathlib.Path(pred_folder)
                if not pred_folder.is_dir():
                    pred_folder.mkdir()
                self.infer_obj.generate_predict_folder(
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
