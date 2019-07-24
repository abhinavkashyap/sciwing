import json
import os
import parsect.constants as constants
from parsect.infer.parsect_inference import ParsectInference
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.datasets.classification.generic_sect_dataset import GenericSectDataset
import torch.nn as nn

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]
FILES = constants.FILES
GENERIC_SECT_LABEL_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


def get_glove_emb_lc_genericsect_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    NUM_CLASSES = config["NUM_CLASSES"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", "random")
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    # it does not matter what random embeddings you have here
    # it will be filled with the learnt parameters while loading the model
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

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECT_LABEL_FILE,
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
    experiment_dirname = os.path.join(OUTPUT_DIR, "debug_bow_glove_genericsect")
    inference_client = get_glove_emb_lc_genericsect_infer(experiment_dirname)
