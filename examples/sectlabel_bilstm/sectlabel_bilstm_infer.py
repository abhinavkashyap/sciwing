import os
import sciwing.constants as constants
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_sectlabel_bilstm_model(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    DATA_PATH = pathlib.Path(DATA_DIR)

    train_file = DATA_PATH.joinpath("sectLabel.train")
    dev_file = DATA_PATH.joinpath("sectLabel.dev")
    test_file = DATA_PATH.joinpath("sectLabel.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    HIDDEN_DIM = 512
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "concat"

    classifier_encoding_dim = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM

    embedder = WordEmbedder(embedding_type="glove_6B_50")

    encoder = LSTM2VecEncoder(
        embedder=embedder,
        hidden_dim=HIDDEN_DIM,
        combine_strategy=COMBINE_STRATEGY,
        bidirectional=BIDIRECTIONAL,
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=classifier_encoding_dim,
        num_classes=23,
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    inference = ClassificationInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return inference


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_sectlabel_bilstm_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
