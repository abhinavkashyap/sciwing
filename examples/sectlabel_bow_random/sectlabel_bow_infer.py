import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


def build_sectlabel_bow_model(dirname: str):
    """

    Parameters
    ----------
    dirname : The directory where sciwing stores your outputs for the model

    Returns
    -------


    """
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

    embedder = WordEmbedder(embedding_type="glove_6B_50")
    encoder = BOW_Encoder(embedder=embedder)
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=embedder.get_embedding_dimension(),
        num_classes=data_manager.num_labels["label"],
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    infer = ClassificationInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )
    return infer


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_sectlabel_bow_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
