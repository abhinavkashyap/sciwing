import sciwing.constants as constants
import pathlib
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
import torch

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_parscit_model(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    data_dir = pathlib.Path(DATA_DIR)
    train_filename = data_dir.joinpath("parscit.train")
    dev_filename = data_dir.joinpath("parscit.dev")
    test_filename = data_dir.joinpath("parscit.test")
    data_manager = SeqLabellingDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
    )

    word_embedder = WordEmbedder(embedding_type="parscit")
    char_embedder = CharEmbedder(
        char_embedding_dimension=25, hidden_dimension=50, datasets_manager=data_manager
    )
    embedder = ConcatEmbedders([word_embedder, char_embedder])

    lstm2seqencoder = Lstm2SeqEncoder(
        embedder=embedder,
        hidden_dim=256,
        bidirectional=True,
        combine_strategy="concat",
        rnn_bias=True,
        device=torch.device("cpu"),
    )

    model = RnnSeqCrfTagger(
        rnn2seqencoder=lstm2seqencoder, encoding_dim=512, datasets_manager=data_manager
    )

    infer = SequenceLabellingInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return infer


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_parscit_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
