from sciwing.infer.interface_client_base import BaseInterfaceClient
from typing import Dict, Any
import wasabi
import sciwing.constants as constants
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.cli.sciwing_interact import SciWINGInteract
import pathlib
from sciwing.infer.seq_label_inference.conll_inference import Conll2003Inference

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


class BuildConllNerSeqCrfInfer(BaseInterfaceClient):
    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = hparams
        data_dir = pathlib.Path(DATA_DIR)
        self.train_filename = data_dir.joinpath("eng.train")
        self.dev_filename = data_dir.joinpath("eng.testa")
        self.test_filename = data_dir.joinpath("eng.testb")
        self.printer = wasabi.Printer()
        self.data_manager = self.build_dataset()
        self.model = self.build_model()
        self.infer = self.build_infer()

    def build_dataset(self):

        data_manager = CoNLLDatasetManager(
            train_filename=self.train_filename,
            dev_filename=self.dev_filename,
            test_filename=self.test_filename,
            column_names=["POS", "DEP", "NER"],
            train_only="ner",
        )

        return data_manager

    def build_model(self):
        embedder = TrainableWordEmbedder(
            embedding_type=self.hparams.get("emb_type"),
            datasets_manager=self.data_manager,
            device=self.hparams.get("device"),
        )

        embedder = ConcatEmbedders([embedder])

        lstm2seqencoder = Lstm2SeqEncoder(
            embedder=embedder,
            dropout_value=self.hparams.get("dropout"),
            hidden_dim=self.hparams.get("hidden_dim"),
            bidirectional=self.hparams.get("bidirectional"),
            combine_strategy=self.hparams.get("combine_strategy"),
            rnn_bias=True,
            device=self.hparams.get("device"),
            num_layers=self.hparams.get("num_layers"),
        )
        model = RnnSeqCrfTagger(
            rnn2seqencoder=lstm2seqencoder,
            encoding_dim=2 * self.hparams.get("hidden_dim")
            if self.hparams.get("bidirectional")
            and self.hparams.get("combine_strategy") == "concat"
            else self.hparams.get("hidden_dim"),
            device=self.hparams.get("device"),
            tagging_type="IOB1",
            datasets_manager=self.data_manager,
        )
        return model

    def build_infer(self):
        infer = Conll2003Inference(
            model=self.model,
            model_filepath=self.hparams.get("model_filepath"),
            datasets_manager=self.data_manager,
        )
        return infer

    def generate_prediction_file(self, output_filename: str):
        self.infer.generate_predictions_for(
            task="ner",
            test_filename=str(self.test_filename),
            output_filename=str(output_filename),
        )


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    model_filepath = dirname.joinpath("checkpoints", "best_model.pt")
    hparams = {
        "emb_type": "glove_6B_100",
        "hidden_dim": 100,
        "bidirectional": False,
        "combine_strategy": "concat",
        "model_filepath": str(model_filepath),
        "device": "cpu",
        "dropout": 0.5,
        "num_layers": 1,
    }
    conll_inference = BuildConllNerSeqCrfInfer(hparams)

    conll_inference.generate_prediction_file(
        output_filename=pathlib.Path("conll_2003_ner_predictions.txt")
    )
