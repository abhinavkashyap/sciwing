import re
import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


def build_genericsect_bow_elmo_model(dirname: str):
    exp_dirpath = pathlib.Path(dirname)

    CUDA_DEVICE = "cpu"

    train_filename = DATA_DIR.joinpath("genericSect.train")
    dev_filename = DATA_DIR.joinpath("genericSect.dev")
    test_filename = DATA_DIR.joinpath("genericSect.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
    )

    embedder = BowElmoEmbedder(
        layer_aggregation="last",
        cuda_device_id=0 if re.match("cuda", CUDA_DEVICE) else -1,
    )

    encoder = BOW_Encoder(embedder=embedder, aggregation_type="sum")
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=1024,
        num_classes=12,
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    parsect_inference = ClassificationInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return parsect_inference


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_genericsect_bow_elmo_model(str(dirname))
    infer.run_inference()
    infer.report_metrics()
