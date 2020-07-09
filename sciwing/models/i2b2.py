from sciwing.utils.vis_seq_tags import VisTagging
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
from sciwing.cli.sciwing_interact import SciWINGInteract
from sciwing.utils.common import cached_path
import sciwing.constants as constants
import pathlib
import json
import torch.nn as nn
import wasabi
from typing import List
from collections import defaultdict

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
DATA_FILE_URLS = constants.DATA_FILE_URLS


class I2B2NER(nn.Module):
    """ It defines a I2B2 clinical NER model trained using SciWING

    For practitioners, we provide ways to obtain results quickly from a set of citations
    stored in a file or from a string. If you want to see the demo head over to our demo site.

    """

    def __init__(self):
        super(I2B2NER, self).__init__()
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)

        if not self.models_cache_dir.is_dir():
            self.models_cache_dir.mkdir(parents=True)

        self.final_model_dir = self.models_cache_dir.joinpath("i2b2")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)

        if not self.data_dir.is_dir():
            self.data_dir.mkdir()

        self.train_data_url = DATA_FILE_URLS["I2B2_TRAIN"]
        self.dev_data_url = DATA_FILE_URLS["I2B2_DEV"]
        self.test_data_url = DATA_FILE_URLS["I2B2_DEV"]
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.hparams = self._get_hparams()
        self.data_manager = self._get_data()
        self.model: nn.Module = self._get_model()
        self.infer = self._get_infer_client()
        self.vis_tagger = VisTagging()
        self.cli_interact = SciWINGInteract(self.infer)

    def _get_model(self) -> nn.Module:
        word_embedder = TrainableWordEmbedder(
            embedding_type=self.hparams.get("emb_type"),
            datasets_manager=self.data_manager,
        )

        elmo_embedder = BowElmoEmbedder(
            datasets_manager=self.data_manager, layer_aggregation="sum"
        )

        embedder = ConcatEmbedders([word_embedder, elmo_embedder])

        lstm2seqencoder = Lstm2SeqEncoder(
            embedder=embedder,
            hidden_dim=self.hparams.get("hidden_dim"),
            bidirectional=self.hparams.get("bidirectional"),
            combine_strategy=self.hparams.get("combine_strategy"),
            rnn_bias=True,
            dropout_value=self.hparams.get("lstm2seq_dropout", 0.0),
            add_projection_layer=False,
        )
        model = RnnSeqCrfTagger(
            rnn2seqencoder=lstm2seqencoder,
            encoding_dim=2 * self.hparams.get("hidden_dim")
            if self.hparams.get("bidirectional")
            and self.hparams.get("combine_strategy") == "concat"
            else self.hparams.get("hidden_dim"),
            datasets_manager=self.data_manager,
        )

        return model

    def _get_infer_client(self):
        infer_client = SequenceLabellingInference(
            model=self.model,
            model_filepath=self.final_model_dir.joinpath("best_model.pt"),
            datasets_manager=self.data_manager,
        )
        return infer_client

    def _predict(self, line: str):
        predictions = self.infer.on_user_input(line=line)
        return predictions

    def predict_for_file(self, filename: str) -> List[str]:
        predictions = defaultdict(list)
        with open(filename, "r") as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                pred_ = self._predict(line=line)
                for namespace, prediction in pred_.items():
                    predictions[namespace].append(prediction[0])
                    stylized_string = self.vis_tagger.visualize_tokens(
                        text=line.split(), labels=prediction[0].split()
                    )
                    self.msg_printer.divider(
                        f"Predictions for Line: {line_idx+1} from {filename}"
                    )
                    print(stylized_string)
                    print("\n")

        return predictions[self.data_manager.label_namespaces[0]]

    def predict_for_text(self, text: str):
        predictions = self._predict(line=text)
        for namespace, prediction in predictions.items():
            self.msg_printer.divider(f"Prediction for {namespace.upper()}")
            stylized_string = self.vis_tagger.visualize_tokens(
                text=text.split(), labels=prediction[0].split()
            )
            print(stylized_string)
            return prediction[0]

    def _get_data(self):
        train_filename = cached_path(
            path=self.data_dir.joinpath("i2b2.train"),
            url=self.train_data_url,
            unzip=False,
        )

        dev_filename = cached_path(
            path=self.data_dir.joinpath("i2b2.dev"), url=self.dev_data_url, unzip=False
        )

        test_filename = cached_path(
            path=self.data_dir.joinpath("i2b2.dev"), url=self.dev_data_url, unzip=False
        )

        data_manager = CoNLLDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            column_names=["NER", "NER", "NER"],
            train_only="ner",
        )
        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        # download the model weights and data to client machine
        cached_path(
            path=f"{self.final_model_dir}.zip",
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/i2b2.zip",
            unzip=True,
        )

    def interact(self):
        self.cli_interact.interact()


if __name__ == "__main__":
    i2b2 = I2B2NER()
    i2b2.interact()
