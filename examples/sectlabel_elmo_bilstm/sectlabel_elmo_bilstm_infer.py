import sciwing.constants as constants
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
import torch
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_sectlabel_elmobilstm_model(dirname: str):
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
    DEVICE = "cpu"
    EMBEDDING_TYPE = "glove_6B_50"
    HIDDEN_DIM = 512
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "concat"

    elmo_embedder = BowElmoEmbedder(
        cuda_device_id=-1 if DEVICE == "cpu" else int(DEVICE.split("cuda:")[1])
    )

    vanilla_embedder = WordEmbedder(embedding_type=EMBEDDING_TYPE)

    embedders = ConcatEmbedders([vanilla_embedder, elmo_embedder])

    encoder = LSTM2VecEncoder(
        embedder=embedders,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        device=torch.device(DEVICE),
    )

    encoding_dim = (
        2 * HIDDEN_DIM if BIDIRECTIONAL and COMBINE_STRATEGY == "concat" else HIDDEN_DIM
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=encoding_dim,
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
    infer = build_sectlabel_elmobilstm_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
