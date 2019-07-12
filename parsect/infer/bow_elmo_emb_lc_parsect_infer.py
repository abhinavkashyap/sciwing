import json
import os
import parsect.constants as constants
from parsect.infer.parsect_inference import ParsectInference
from parsect.models.bow_elmo_linear_classifier import BowElmoLinearClassifier
from parsect.modules.bow_elmo_encoder import BowElmoEncoder
from parsect.datasets.parsect_dataset import ParsectDataset

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_elmo_emb_lc_infer_parsect(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", "random")
    NUM_CLASSES = config["NUM_CLASSES"]

    encoder = BowElmoEncoder(emb_dim=EMBEDDING_DIM)

    model = BowElmoLinearClassifier(
        encoder=encoder, encoding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIM,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_emb_lc_parsect")
    inference_client = get_elmo_emb_lc_infer_parsect(experiment_dirname)
