import parsect.constants as constants
import os
from parsect.models.elmo_lstm_classifier import ElmoLSTMClassifier
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
from parsect.modules.elmo_embedder import ElmoEmbedder
from parsect.clients.parsect_inference import ParsectInference
import json
import torch
import torch.nn as nn

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_elmo_bilstm_lc_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
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
        device=torch.device(DEVICE)
    )
    encoding_dim = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM

    model = ElmoLSTMClassifier(
        elmo_lstm_encoder=elmo_lstm_encoder,
        encoding_dim=encoding_dim,
        num_classes=NUM_CLASSES,
        device=torch.device(DEVICE)
    )

    inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath
    )
    return inference


if __name__ == '__main__':
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "debug_elmo_bi_lstm_lc"
    )
    inference_client = get_elmo_bilstm_lc_infer(experiment_dirname)
