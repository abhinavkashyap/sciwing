import os
import sciwing.constants as constants
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.infer.interface_client_base import BaseInterfaceClient
from sciwing.cli.sciwing_interact import SciWINGInteract
from typing import Dict, Any
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


class BuildGenericSectBiLSTMInfer(BaseInterfaceClient):
    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = hparams
        self.data_manager = self.build_dataset()
        self.model = self.build_model()

    def build_model(self):
        embedder = WordEmbedder(embedding_type=self.hparams.get("embedding_type"))

        encoder = LSTM2VecEncoder(
            embedder=embedder,
            hidden_dim=self.hparams.get("hidden_dim"),
            combine_strategy=self.hparams.get("combine_strategy"),
            bidirectional=self.hparams.get("bidirectional"),
        )

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=2 * self.hparams.get("hidden_dim"),
            num_classes=self.hparams.get("num_classes"),
            classification_layer_bias=True,
            datasets_manager=self.data_manager,
        )

        return model

    def build_dataset(self):
        train_filename = DATA_DIR.joinpath("genericSect.train")
        dev_filename = DATA_DIR.joinpath("genericSect.dev")
        test_filename = DATA_DIR.joinpath("genericSect.test")

        data_manager = TextClassificationDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
        )
        return data_manager

    def build_infer(self):
        inference = ClassificationInference(
            model=self.model,
            model_filepath=self.hparams.get("model_filepath"),
            datasets_manager=self.data_manager,
        )

        return inference


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    model_filepath = dirname.joinpath("checkpoints", "best_model.pt")
    hparams = {
        "embedding_type": "glove_6B_50",
        "hidden_dim": 512,
        "bidirectional": True,
        "combine_strategy": "concat",
        "num_classes": 12,
        "model_filepath": model_filepath,
    }
    infer = BuildGenericSectBiLSTMInfer(hparams=hparams)
    cli = SciWINGInteract(infer_client=infer)
    cli.interact()
