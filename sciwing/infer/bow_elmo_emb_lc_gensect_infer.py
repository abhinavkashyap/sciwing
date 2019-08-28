import json
import os
import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.datasets.classification.generic_sect_dataset import GenericSectDataset
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_elmo_emb_lc_infer_gensect(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]

    NUM_CLASSES = config["NUM_CLASSES"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    LAYER_AGGREGATION = config["LAYER_AGGREGATION"]
    WORD_AGGREGATION = config["WORD_AGGREGATION"]

    embedder = BowElmoEmbedder(
        emb_dim=EMBEDDING_DIMENSION, layer_aggregation=LAYER_AGGREGATION
    )
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        aggregation_type=WORD_AGGREGATION,
    )
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = GenericSectDataset(**test_dataset_args)

    parsect_inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_gensect")
    inference_client = get_elmo_emb_lc_infer_gensect(experiment_dirname)
