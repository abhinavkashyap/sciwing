import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.infer.interface_client_base import BaseInterfaceClient
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.cli.sciwing_interact import SciWINGInteract

import pathlib
from typing import Dict, Any

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


class BuildGenericSectBowRandom(BaseInterfaceClient):
    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = hparams
        self.data_manager = self.build_dataset()
        self.model = self.build_model()

    def build_model(self):
        embedder = WordEmbedder(embedding_type=self.hparams.get("emb_type"))
        encoder = BOW_Encoder(embedder=embedder)

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=self.hparams.get("encoding_dim"),
            num_classes=self.hparams.get("num_classes"),
            classification_layer_bias=True,
            datasets_manager=self.data_manager,
        )

        return model

    def build_dataset(self):
        data_dir = pathlib.Path(DATA_DIR)
        train_filename = data_dir.joinpath("genericSect.train")
        dev_filename = data_dir.joinpath("genericSect.dev")
        test_filename = data_dir.joinpath("genericSect.test")

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
        "emb_type": "glove_6B_50",
        "model_filepath": str(model_filepath),
        "num_classes": 12,
        "encoding_dim": 50,
    }
    sectlabel_infer = BuildGenericSectBowRandom(hparams)
    cli = SciWINGInteract(sectlabel_infer)
    cli.interact()
