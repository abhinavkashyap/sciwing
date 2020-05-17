import sciwing.constants as constants
from sciwing.datasets import CoNLLDatasetManager
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.infer.seq_label_inference.seq_label_inference import SequenceLabellingInference
from sciwing.models import RnnSeqCrfTagger
from sciwing.modules import CharEmbedder, Lstm2SeqEncoder
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.utils.common import cached_path
import pathlib
import json
import wasabi
from typing import List
import torch



PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class ScienceIE:
    def __init__(self):
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("lstm_crf_scienceie_final")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()

    def _get_model(self):
        word_embedder = TrainableWordEmbedder(
            embedding_type=self.hparams.get("emb_type"), datasets_manager=self.data_manager
        )
        char_embedder = CharEmbedder(
            char_embedding_dimension=self.hparams.get("char_emb_dim"), hidden_dimension=self.hparams.get("char_encoder_hidden_dim"), datasets_manager=self.data_manager
        )

        # concat the embeddings
        embedder = ConcatEmbedders([char_embedder, word_embedder])
        lstm2seqencoder = Lstm2SeqEncoder(
            embedder=embedder,
            hidden_dim=self.hparams.get("hidden_dim"),
            bidirectional = self.hparams.get("bidirectional"),
            combine_strategy=self.hparams.get("combine_strategy"),
            rnn_bias=True,
            device=torch.device(self.hparams.get("device")),
            num_layers=self.hparams.get("num_layers"),
        )

        model = RnnSeqCrfTagger(
            rnn2seqencoder=lstm2seqencoder,
            encoding_dim=2 * self.hparams.get("hidden_dim")
            if self.hparams.get("bidirectional") and self.hparams.get("combine_strategy") == "concat"
            else self.hparams.get("hidden_dim"),
            tagging_type="BIOUL",
            namespace_to_constraints=None,
            datasets_manager=self.data_manager,
        )
        return model

    def _get_infer_client(self):
        client = SequenceLabellingInference(
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
        train_filename = self.data_dir.joinpath("train_science_ie_conll.txt")
        dev_filename = self.data_dir.joinpath("dev_science_ie_conll.txt")
        test_filename = self.data_dir.joinpath("dev_science_ie_conll.txt")

        data_manager = CoNLLDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            column_names=["TASK", "PROCESS", "MATERIAL"],

        )

        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        cached_path(
            path=self.final_model_dir,
            url="https://science-ie-models.s3-ap-southeast-1.amazonaws.com/lstm_crf_scienceie_final.zip",
        )
