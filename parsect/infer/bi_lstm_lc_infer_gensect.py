import parsect
import os
import parsect.constants as constants
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.clients.parsect_inference import ParsectInference
from parsect.datasets.generic_sect_dataset import GenericSectDataset
import json
import torch.nn as nn

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


def get_bilstm_lc_infer_gensect(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_TYPE = config["EMBEDDING_TYPE"]
    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    HIDDEN_DIM = config["HIDDEN_DIMENSION"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    classifier_encoding_dim = 2 * HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM

    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    encoder = LSTM2VecEncoder(
        emb_dim=EMBEDDING_DIM,
        embedding=embedding,
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

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIM,
    )

    inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
    )

    return inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bi_lstm_lc_gensect")
    inference_client = get_bilstm_lc_infer_gensect(experiment_dirname)
