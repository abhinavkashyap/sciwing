import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_sectlabel_bow_elmo_model(dirname: str):
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

    embedder = BowElmoEmbedder(layer_aggregation="last")
    encoder = BOW_Encoder(aggregation_type="sum", embedder=embedder)

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=1024,
        num_classes=data_manager.num_labels["label"],
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    infer_client = ClassificationInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return infer_client


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_sectlabel_bow_elmo_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
