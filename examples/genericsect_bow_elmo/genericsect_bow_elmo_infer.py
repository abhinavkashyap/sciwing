import re
import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.infer.interface_client_base import BaseInterfaceClient
from sciwing.cli.sciwing_interact import SciWINGInteract
import pathlib
from typing import Dict, Any

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


class BuildGenericSectBowElmo(BaseInterfaceClient):
    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = hparams
        self.data_manager = self.build_dataset()
        self.model = self.build_model()

    def build_model(self):
        embedder = BowElmoEmbedder(
            layer_aggregation=self.hparams.get("layer_aggregation")
        )

        encoder = BOW_Encoder(
            aggregation_type=self.hparams.get("word_aggregation"), embedder=embedder
        )

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=self.hparams.get("encoding_dim"),
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
        parsect_inference = ClassificationInference(
            model=self.model,
            model_filepath=self.hparams.get("model_filepath"),
            datasets_manager=self.data_manager,
        )
        return parsect_inference


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    model_filepath = dirname.joinpath("checkpoints", "best_model.pt")
    hparams = {
        "layer_aggregation": "last",
        "word_aggregation": "sum",
        "encoding_dim": 1024,
        "num_classes": 12,
        "model_filepath": model_filepath,
    }
    infer = BuildGenericSectBowElmo(hparams)
    cli = SciWINGInteract(infer)
    cli.interact()
