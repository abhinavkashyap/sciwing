import sciwing.constants as constants
import pathlib
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
from sciwing.infer.interface_client_base import BaseInterfaceClient
from sciwing.cli.sciwing_interact import SciWINGInteract
from typing import Dict, Any
import wasabi

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


class BuildParscitInterference(BaseInterfaceClient):
    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = hparams
        self.printer = wasabi.Printer()
        self.data_manager = self.build_dataset()
        self.model = self.build_model()

    def build_model(self):

        word_embedder = TrainableWordEmbedder(
            embedding_type=self.hparams.get("emb_type"),
            datasets_manager=self.data_manager,
        )

        lstm2seqencoder = Lstm2SeqEncoder(
            embedder=word_embedder,
            hidden_dim=self.hparams.get("hidden_dim"),
            bidirectional=self.hparams.get("bidirectional"),
            combine_strategy=self.hparams.get("combine_strategy"),
            rnn_bias=True,
            device=self.hparams.get("device"),
        )

        model = RnnSeqCrfTagger(
            rnn2seqencoder=lstm2seqencoder,
            encoding_dim=2 * self.hparams.get("hidden_dim")
            if self.hparams.get("bidirectional")
            and self.hparams.get("combine_strategy") == "concat"
            else self.hparams.get("hidden_dim"),
            datasets_manager=self.data_manager,
        )

        self.printer.good("Finished Loading the Model")
        return model

    def build_dataset(self):
        data_dir = pathlib.Path(DATA_DIR)
        train_filename = data_dir.joinpath("parscit.train")
        dev_filename = data_dir.joinpath("parscit.dev")
        test_filename = data_dir.joinpath("parscit.test")
        data_manager = SeqLabellingDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
        )
        return data_manager

    def build_infer(self):
        infer = SequenceLabellingInference(
            model=self.model,
            model_filepath=self.hparams.get("model_filepath"),
            datasets_manager=self.data_manager,
        )
        return infer


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    model_filepath = dirname.joinpath("checkpoints", "best_model.pt")
    hparams = {
        "emb_type": "glove_6B_100",
        "char_emb_dim": 25,
        "char_encoder_hidden_dim": 100,
        "hidden_dim": 256,
        "bidirectional": True,
        "combine_strategy": "concat",
        "device": "cpu",
        "model_filepath": str(model_filepath),
    }
    parscit_inference = BuildParscitInterference(hparams)
    cli = SciWINGInteract(parscit_inference)
    cli.interact()
