import os
import sciwing.constants as constants
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


def build_genericsect_bilstm_model(dirname: str):

    exp_dirpath = pathlib.Path(dirname)

    EMBEDDING_TYPE = "glove_6B_50"
    HIDDEN_DIM = 512
    COMBINE_STRATEGY = "concat"
    BIDIRECTIONAL = True

    train_filename = DATA_DIR.joinpath("genericSect.train")
    dev_filename = DATA_DIR.joinpath("genericSect.dev")
    test_filename = DATA_DIR.joinpath("genericSect.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
    )

    embedder = WordEmbedder(embedding_type=EMBEDDING_TYPE)

    encoder = LSTM2VecEncoder(
        embedder=embedder,
        hidden_dim=HIDDEN_DIM,
        combine_strategy=COMBINE_STRATEGY,
        bidirectional=BIDIRECTIONAL,
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=2 * HIDDEN_DIM,
        num_classes=12,
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
    infer = build_genericsect_bilstm_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
