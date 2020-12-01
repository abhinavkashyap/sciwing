import questionary
from questionary import Choice
import pathlib
from sciwing.utils.amazon_s3 import S3Util
import sciwing.constants as constants
import os
import wasabi
import shutil

PATHS = constants.PATHS
AWS_CRED_DIR = PATHS["AWS_CRED_DIR"]
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]


class S3OutputMove:
    def __init__(self, foldername: str):
        """ Provides an interactive way to move some folders to s3

        Parameters
        ----------
        foldername : str
            The folder name which will be moved to S3 bucket
        """
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
        """ Goes through the folder and gets the choice on which folder should be moved

        Returns
        -------
        str
            The folder which is chosen to be moved

        """
        choices = []
        path = pathlib.Path(self.foldername)

        for dir in path.iterdir():
            choices.append(Choice(str(dir)))

        choices.append(Choice("exit"))

        folder_chose_question = questionary.select(
            "Folder in the directory. Chose one to move to s3",
            qmark="❓",
            choices=choices,
        )
        folder_type_answer = folder_chose_question.ask()

        return folder_type_answer

    def interact(self):
        """ Interacts with the user by providing various options
        """
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
        """ Since this is deletion, we want confirmation, just to be sure
        whether to keep the deleted folder locally or to remove it

        Returns
        -------
        str
            An yes or no answer to the question

        """
        deletion_question = questionary.rawselect(
            "Do you also want to delete the file locally. Caution! File will be removed locally",
            qmark="❓",
            choices=[Choice("yes"), Choice("no")],
        )
        deletion_answer = deletion_question.ask()
        return deletion_answer


if __name__ == "__main__":
    s3_move_cli = S3OutputMove(foldername=MODELS_CACHE_DIR)
