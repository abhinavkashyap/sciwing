import json
import os
import sciwing.constants as constants
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.embedders.vanilla_embedder import WordEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.datasets.classification.generic_sect_dataset import GenericSectDataset
import torch.nn as nn
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


def get_bow_lc_gensect_infer(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
    embedder = WordEmbedder(embedding_dim=EMBEDDING_DIMENSION, embedding=embedding)
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        dropout_value=0.0,
        aggregation_type="sum",
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = GenericSectDataset(**test_dataset_args)

    parsect_inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return parsect_inference
