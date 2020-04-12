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
from sciwing.utils.common import cached_path
import pathlib
import json
import wasabi
from typing import List


PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class GenericSect:
    def __init__(self):
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("genericsect_bow_elmo")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()

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
        lines = []
        with open(filename) as fp:
            for line in fp:
                lines.append(line)

        predictions = self.infer.infer_batch(lines=lines)
        for line, prediction in zip(lines, predictions):
            self.msg_printer.text(title=line, text=prediction)

        return predictions

    def predict_for_text(self, text: str) -> str:
        prediction = self.infer.on_user_input(line=text)
        self.msg_printer.text(title=text, text=prediction)
        return prediction

    def _get_data(self):
        train_filename = self.data_dir.joinpath("genericSect.train")
        dev_filename = self.data_dir.joinpath("genericSect.dev")
        test_filename = self.data_dir.joinpath("genericSect.test")

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
            path=self.final_model_dir,
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/genericsect_bow_elmo.zip",
        )
