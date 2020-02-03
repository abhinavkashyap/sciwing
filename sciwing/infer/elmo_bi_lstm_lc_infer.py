import sciwing.constants as constants
import os
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.vanilla_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.datasets.classification.sectlabel_dataset import SectLabelDataset
from sciwing.infer.classification.classification_inference import (
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
    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    HIDDEN_DIM = config["HIDDEN_DIMENSION"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    NUM_CLASSES = config["NUM_CLASSES"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    elmo_embedder = BowElmoEmbedder(
        layer_aggregation="sum",
        cuda_device_id=-1 if DEVICE == "cpu" else int(DEVICE.split("cuda:")[1]),
    )

    vanilla_embedder = WordEmbedder(embedding=embedding, embedding_dim=EMBEDDING_DIM)

    embedders = ConcatEmbedders([vanilla_embedder, elmo_embedder])

    encoder = LSTM2VecEncoder(
        emb_dim=EMBEDDING_DIM + 1024,
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
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = SectLabelDataset(**test_dataset_args)

    inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )
    return inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_parsect_elmo_bi_lstm_lc")
    inference_client = get_elmo_bilstm_lc_infer(experiment_dirname)
