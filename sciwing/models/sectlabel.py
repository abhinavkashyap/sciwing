import sciwing.constants as constants
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
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


class SectLabel:
    def __init__(self):
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("sectlabel_elmo_bilstm")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()

    def _get_model(self):
        elmo_embedder = BowElmoEmbedder(layer_aggregation="sum")

        # instantiate the vanilla embedder
        vanilla_embedder = WordEmbedder(embedding_type=self.hparams.get("emb_type"))

        # concat the embeddings
        embedder = ConcatEmbedders([vanilla_embedder, elmo_embedder])

        hidden_dim = self.hparams.get("hidden_dim")
        bidirectional = self.hparams.get("bidirectional")
        combine_strategy = self.hparams.get("combine_strategy")

        encoder = LSTM2VecEncoder(
            embedder=embedder,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            combine_strategy=combine_strategy,
        )

        encoding_dim = (
            2 * hidden_dim
            if bidirectional and combine_strategy == "concat"
            else hidden_dim
        )

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=encoding_dim,
            num_classes=23,
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
        train_filename = self.data_dir.joinpath("sectLabel.train")
        dev_filename = self.data_dir.joinpath("sectLabel.dev")
        test_filename = self.data_dir.joinpath("sectLabel.test")

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
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/sectlabel_elmo_bilstm.zip",
        )
