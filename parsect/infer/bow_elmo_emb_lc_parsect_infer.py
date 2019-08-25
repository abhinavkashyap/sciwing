import json
import os
import parsect.constants as constants
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
from parsect.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.classification.parsect_dataset import ParsectDataset

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
    LAYER_AGGREGATION = config.get("LAYER_AGGREGATION")
    WORD_AGGREGATION = config.get("WORD_AGGREGATION")

    embedder = BowElmoEmbedder(
        emb_dim=EMBEDDING_DIM, layer_aggregation=LAYER_AGGREGATION
    )
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIM, aggregation_type=WORD_AGGREGATION, embedder=embedder
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    dataset = ParsectDataset(
        filename=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIM,
    )

    parsect_inference = ClassificationInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
        dataset_class=ParsectDataset,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_elmo_emb_lc_parsect")
    inference_client = get_elmo_emb_lc_infer_parsect(experiment_dirname)
