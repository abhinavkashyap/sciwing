import os
import sciwing.constants as constants
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.datasets.classification.sectlabel_dataset import SectLabelDataset
import json
import torch.nn as nn
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_bilstm_lc_infer_parsect(dirname: str):

    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    HIDDEN_DIM = config["HIDDEN_DIMENSION"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    classifier_encoding_dim = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    embedder = WordEmbedder(embedding_dim=EMBEDDING_DIM, embedding=embedding)

    encoder = LSTM2VecEncoder(
        emb_dim=EMBEDDING_DIM,
        embedder=embedder,
        hidden_dim=HIDDEN_DIM,
        combine_strategy=COMBINE_STRATEGY,
        bidirectional=BIDIRECTIONAL,
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=classifier_encoding_dim,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = SectLabelDataset(**test_dataset_args)

    inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bi_lstm_lc")
    inference_client = get_bilstm_lc_infer_parsect(experiment_dirname)
