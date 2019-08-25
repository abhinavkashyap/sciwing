import json
import os
import parsect.constants as constants
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.datasets.classification.generic_sect_dataset import GenericSectDataset
from parsect.modules.embedders.bow_elmo_embedder import BowElmoEmbedder

PATHS = constants.PATHS
FILES = constants.FILES
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_elmo_emb_lc_infer_gensect(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]

    NUM_CLASSES = config["NUM_CLASSES"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", "random")
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    LAYER_AGGREGATION = config["LAYER_AGGREGATION"]
    WORD_AGGREGATION = config["WORD_AGGREGATION"]

    embedder = BowElmoEmbedder(
        emb_dim=EMBEDDING_DIMENSION, layer_aggregation=LAYER_AGGREGATION
    )
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        aggregation_type=WORD_AGGREGATION,
    )
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    parsect_inference = ClassificationInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
        dataset_class=GenericSectDataset,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_gensect")
    inference_client = get_elmo_emb_lc_infer_gensect(experiment_dirname)
