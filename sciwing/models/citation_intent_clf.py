import torch.nn as nn
from typing import List
import sciwing.constants as constants
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.elmo_embedder import ElmoEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.utils.common import cached_path
import pathlib
import wasabi
import json

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class CitationIntentClassification(nn.Module):
    def __init__(self):
        super(CitationIntentClassification, self).__init__()
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath(
            "citation_intent_clf_elmo", "checkpoints"
        )
        self.data_dir = pathlib.Path(DATA_DIR)
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.hparams = self._get_hparams()
        self.data_manager = self._get_data()
        self.model: nn.Module = self._get_model()
        self.infer = self._get_infer_client()

    def _get_model(self) -> nn.Module:
        embedding_type = self.hparams.get("emb_type")
        word_embedder = WordEmbedder(embedding_type=embedding_type)
        elmo_embedder = ElmoEmbedder(datasets_manager=self.data_manager)
        embedder = ConcatEmbedders([word_embedder, elmo_embedder])

        hidden_dim = self.hparams.get("hidden_dim")
        combine_strategy = self.hparams.get("combine_strategy")
        bidirectional = self.hparams.get("bidirectional")

        encoder = LSTM2VecEncoder(
            embedder=embedder,
            hidden_dim=hidden_dim,
            combine_strategy=combine_strategy,
            bidirectional=bidirectional,
        )

        classifier_encoding_dim = 2 * hidden_dim if bidirectional else hidden_dim
        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=classifier_encoding_dim,
            num_classes=3,
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
        with open(filename, "r") as fp:
            lines = []
            for line in fp:
                line = line.strip()
                lines.append(line)

            predictions = self.infer.infer_batch(lines=lines)
            for prediction, line in zip(predictions, lines):
                self.msg_printer.text(title=line, text=prediction)

        return predictions

    def predict_for_text(self, text: str):
        label = self.infer.on_user_input(line=text)
        self.msg_printer.text(title=text, text=label)
        return label

    def _get_data(self):
        data_manager = TextClassificationDatasetManager(
            train_filename=self.data_dir.joinpath("scicite.train"),
            dev_filename=self.data_dir.joinpath("scicite.dev"),
            test_filename=self.data_dir.joinpath("scicite.test"),
        )
        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        # download the model weights and data to client machine
        cached_path(
            path=self.final_model_dir,
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/citation_intent_clf_elmo.zip",
        )
