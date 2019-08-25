import parsect.constants as constants
import os
from parsect.models.elmo_lstm_classifier import ElmoLSTMClassifier
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
from parsect.modules.embedders.elmo_embedder import ElmoEmbedder
from parsect.datasets.classification.parsect_dataset import ParsectDataset
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
import json
import torch
import torch.nn as nn
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_elmo_bilstm_lc_infer(dirname: str):

    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    DEVICE = config["DEVICE"]
    ELMO_EMBEDDING_DIMENSION = config["ELMO_EMBEDDING_DIMENSION"]
    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    HIDDEN_DIM = config["HIDDEN_DIMENSION"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    NUM_CLASSES = config["NUM_CLASSES"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    elmo_embedder = ElmoEmbedder(device=torch.device(DEVICE))
    elmo_lstm_encoder = ElmoLSTMEncoder(
        elmo_emb_dim=ELMO_EMBEDDING_DIMENSION,
        elmo_embedder=elmo_embedder,
        emb_dim=EMBEDDING_DIM,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        device=torch.device(DEVICE),
    )
    encoding_dim = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM

    model = ElmoLSTMClassifier(
        elmo_lstm_encoder=elmo_lstm_encoder,
        encoding_dim=encoding_dim,
        num_classes=NUM_CLASSES,
        device=torch.device(DEVICE),
    )

    dataset = ParsectDataset(**test_dataset_args)

    inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )
    return inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_elmo_bi_lstm_lc")
    inference_client = get_elmo_bilstm_lc_infer(experiment_dirname)
