import sciwing.constants as constants
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.cli.sciwing_interact import SciWINGInteract
from sciwing.utils.common import cached_path
import pathlib
import json
import wasabi
from typing import List


PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
DATA_FILE_URLS = constants.DATA_FILE_URLS


class GenericSect:
    def __init__(self):
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("genericsect_bow_elmo")

        if not self.models_cache_dir.is_dir():
            self.models_cache_dir.mkdir(parents=True)

        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)

        if not self.data_dir.is_dir():
            self.data_dir.mkdir(parents=True)

        self.train_data_url = DATA_FILE_URLS["GENERIC_SECTION_TRAIN_FILE"]
        self.dev_data_url = DATA_FILE_URLS["GENERIC_SECTION_DEV_FILE"]
        self.test_data_url = DATA_FILE_URLS["GENERIC_SECTION_TEST_FILE"]

        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()
        self.cli_interact = SciWINGInteract(self.infer)

    def _get_model(self):
        embedder = BowElmoEmbedder(
            layer_aggregation=self.hparams.get("layer_aggregation"),
            datasets_manager=self.data_manager,
        )
        encoder = BOW_Encoder(
            aggregation_type=self.hparams.get("word_aggregation"), embedder=embedder
        )

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=1024,
            num_classes=12,
            classification_layer_bias=True,
            datasets_manager=self.data_manager,
        )
        return model

    def _get_infer_client(self):
        client = ClassificationInference(
            model=self.model,
            model_filepath=self.final_model_dir.joinpath("best_model.pt"),
            datasets_manager=self.data_manager,
        )
        return client

    def predict_for_file(self, filename: str) -> List[str]:
        """ Make predictions for every line in the file

        Parameters
        ----------
        filename: str
            The filename where section headers are stored one per line

        Returns
        -------
        List[str]
            A list of predictions

        """
        lines = []
        with open(filename) as fp:
            for line in fp:
                lines.append(line)

        predictions = self.infer.infer_batch(lines=lines)
        for line, prediction in zip(lines, predictions):
            self.msg_printer.text(title=line, text=prediction)

        return predictions

    def predict_for_text(self, text: str, show=True) -> str:
        """ Predicts the generic section headers of the text

        Parameters
        ----------
        text: str
            The section header string to be normalized
        show : bool
            If True then we print the prediction.

        Returns
        -------
        str
            The prediction for the section header

        """
        prediction = self.infer.on_user_input(line=text)
        if show:
            self.msg_printer.text(title=text, text=prediction)
        return prediction

    def _get_data(self):
        train_filename = self.data_dir.joinpath("genericSect.train")
        dev_filename = self.data_dir.joinpath("genericSect.dev")
        test_filename = self.data_dir.joinpath("genericSect.test")

        train_filename = cached_path(
            path=train_filename, url=self.train_data_url, unzip=False
        )

        dev_filename = cached_path(
            path=dev_filename, url=self.dev_data_url, unzip=False
        )

        test_filename = cached_path(
            path=test_filename, url=self.test_data_url, unzip=False
        )

        data_manager = TextClassificationDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
        )

        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        cached_path(
            path=f"{self.final_model_dir}.zip",
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/genericsect_bow_elmo.zip",
            unzip=True,
        )

    def interact(self):
        """ Interact with the pretrained model
        """
        self.cli_interact.interact()


if __name__ == "__main__":
    generic_sect = GenericSect()
    generic_sect.interact()
