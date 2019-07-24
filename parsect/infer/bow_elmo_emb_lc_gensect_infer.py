import json
import os
import parsect.constants as constants
from parsect.infer.parsect_inference import ParsectInference
from parsect.models.bow_elmo_linear_classifier import BowElmoLinearClassifier
from parsect.datasets.classification.generic_sect_dataset import GenericSectDataset
from parsect.modules.bow_elmo_encoder import BowElmoEncoder

PATHS = constants.PATHS
FILES = constants.FILES
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_elmo_emb_lc_infer_gensect(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    encoder = BowElmoEncoder(emb_dim=EMBEDDING_DIM)

    NUM_CLASSES = config["NUM_CLASSES"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", "random")
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]

    model = BowElmoLinearClassifier(
        encoder=encoder, encoding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_gensect")
    inference_client = get_elmo_emb_lc_infer_gensect(experiment_dirname)
