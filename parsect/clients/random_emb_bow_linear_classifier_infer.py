import json
import os
import parsect.constants as constants
from parsect.clients.parsect_inference import ParsectInference
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.modules.bow_encoder import BOW_Encoder
import torch.nn as nn

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]


def get_random_emb_linear_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EXP_NAME = config["EXP_NAME"]
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")

    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    SAVE_EVERY = config["SAVE_EVERY"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)

    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embedding,
        dropout_value=0.0,
        aggregation_type="sum",
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
    )

    return parsect_inference
