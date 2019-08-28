import json
import os
import sciwing.constants as constants
from sciwing.modules.embedders.bert_embedder import BertEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.generic_sect_dataset import GenericSectDataset
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
import pathlib

import torch

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


def get_bow_bert_emb_lc_gensect_infer(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    BERT_TYPE = config["BERT_TYPE"]

    DEVICE = config["DEVICE"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedder = BertEmbedder(
        emb_dim=EMBEDDING_DIM,
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE),
    )

    encoder = BOW_Encoder(
        embedder=embedder, emb_dim=EMBEDDING_DIM, aggregation_type="average"
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = GenericSectDataset(**test_dataset_args)

    parsect_inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return parsect_inference
