import questionary
from questionary import Choice
import pathlib
from parsect.utils.amazon_s3 import S3Util
import parsect.constants as constants
import os
import wasabi
import shutil

PATHS = constants.PATHS
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


class S3OutputMove:
    """
    This provides an option to move the model folders to s3
    You can also delete the folder locally once its moved to s3
    """

    def __init__(self, foldername: str):
        self.foldername = foldername
        self.s3_config_json_filename = os.path.join(
            AWS_CRED_DIR, "aws_s3_credentials.json"
        )
        self.s3_util = S3Util(
            aws_cred_config_json_filename=self.s3_config_json_filename
        )
        self.msg_printer = wasabi.Printer()
        self.interact()

    def get_folder_choice(self):
        choices = []
        path = pathlib.Path(self.foldername)

        for dir in path.iterdir():
            choices.append(Choice(str(dir)))

        choices.append(Choice("exit"))

        folder_chose_question = questionary.select(
            "These experiments exist in the output folder. Chose " "one to move to s3",
            qmark="❓",
            choices=choices,
        )
        folder_type_answer = folder_chose_question.ask()

        return folder_type_answer

    def interact(self):
        while True:
            answer = self.get_folder_choice()
            if answer == "exit":
                break
            else:
                with self.msg_printer.loading(f"Uploading {answer} to s3"):
                    folder_name = answer
                    base_folder_name = pathlib.Path(answer).name
                    self.s3_util.upload_folder(
                        folder_name=answer, base_folder_name=base_folder_name
                    )
                self.msg_printer.good(f"Moved folder {answer} to s3")
                deletion_answer = self.ask_deletion()
                if deletion_answer == "yes":
                    folder_path = pathlib.Path(answer)
                    shutil.rmtree(folder_path)

    @staticmethod
    def ask_deletion() -> str:
        deletion_question = questionary.rawselect(
            "Do you also want to delete the file locally. Caution! File will be removed locally",
            qmark="❓",
            choices=[Choice("yes"), Choice("no")],
        )
        deletion_answer = deletion_question.ask()
        return deletion_answer


if __name__ == "__main__":
    s3_move_cli = S3OutputMove(foldername=OUTPUT_DIR)
