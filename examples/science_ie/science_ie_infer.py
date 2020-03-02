import sciwing.constants as constants
import pathlib
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
from sciwing.utils.science_ie_eval import calculateMeasures
import torch

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_science_ie_model(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    data_dir = pathlib.Path(DATA_DIR)
    train_filename = data_dir.joinpath("train_science_ie_conll.txt")
    dev_filename = data_dir.joinpath("dev_science_ie_conll.txt")
    test_filename = data_dir.joinpath("dev_science_ie_conll.txt")
    data_manager = CoNLLDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
        column_names=["TASK", "PROCESS", "MATERIAL"],
    )

    word_embedder = TrainableWordEmbedder(
        embedding_type="glove_6B_100", datasets_manager=data_manager
    )
    char_embedder = CharEmbedder(
        char_embedding_dimension=50, hidden_dimension=100, datasets_manager=data_manager
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
        rnn2seqencoder=lstm2seqencoder,
        encoding_dim=512,
        datasets_manager=data_manager,
        namespace_to_constraints=None,
        tagging_type="BIOUL",
    )

    infer = SequenceLabellingInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return infer


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_science_ie_model(str(dirname))
    # infer.run_inference()
    # infer.report_metrics()
    prediction_folder = pathlib.Path(".", "science_ie_pred")
    FILES = constants.FILES
    SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]
    SCIENCE_IE_DEV_FOLDER = pathlib.Path(SCIENCE_IE_DEV_FOLDER)
    if not prediction_folder.is_dir():
        prediction_folder.mkdir()

    # infer.generate_scienceie_prediction_folder(
    #     dev_folder=SCIENCE_IE_DEV_FOLDER, pred_folder=prediction_folder
    # )

    calculateMeasures(
        folder_gold=str(SCIENCE_IE_DEV_FOLDER),
        folder_pred=str(prediction_folder),
        remove_anno="rel",
    )
