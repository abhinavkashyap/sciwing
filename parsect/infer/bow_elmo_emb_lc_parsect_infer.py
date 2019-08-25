import json
import os
import parsect.constants as constants
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
from parsect.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.classification.parsect_dataset import ParsectDataset
import pathlib

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_elmo_emb_lc_infer_parsect(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    LAYER_AGGREGATION = config.get("LAYER_AGGREGATION")
    WORD_AGGREGATION = config.get("WORD_AGGREGATION")

    embedder = BowElmoEmbedder(
        emb_dim=EMBEDDING_DIM, layer_aggregation=LAYER_AGGREGATION
    )
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIM, aggregation_type=WORD_AGGREGATION, embedder=embedder
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = ParsectDataset(**test_dataset_args)

    parsect_inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_emb_lc_parsect")
    inference_client = get_elmo_emb_lc_infer_parsect(experiment_dirname)
