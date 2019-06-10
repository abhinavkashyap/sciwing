import parsect
import os
import parsect.constants as constants
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.clients.parsect_inference import ParsectInference
import json
import torch.nn as nn

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_bilstm_lc_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

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

    encoder = LSTM2VecEncoder(
        emb_dim=EMBEDDING_DIM,
        embedding=embedding,
        hidden_dim=HIDDEN_DIM,
        combine_strategy=COMBINE_STRATEGY,
        bidirectional=BIDIRECTIONAL
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=classifier_encoding_dim,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True
    )

    inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath
    )

    return inference


if __name__ == '__main__':
    experiment_dirname = os.path.join(OUTPUT_DIR, 'debug_bi_lstm_lc')
    inference_client = get_bilstm_lc_infer(experiment_dirname)
