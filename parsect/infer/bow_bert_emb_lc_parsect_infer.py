import json
import os
import parsect.constants as constants
from parsect.modules.bow_bert_encoder import BowBertEncoder
from parsect.models.bow_bert_linear_classifier import BowBertLinearClassifier
from parsect.datasets.parsect_dataset import ParsectDataset
from parsect.infer.parsect_inference import ParsectInference
import torch

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_bow_bert_emb_lc_parsect_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    BERT_TYPE = config["BERT_TYPE"]

    DEVICE = config["DEVICE"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", "random")
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    vocab_store_location = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    encoder = BowBertEncoder(
        emb_dim=EMBEDDING_DIM,
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE),
    )

    model = BowBertLinearClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "debug_bow_bert_base_cased_emb_lc_10e_1e-2lr"
    )
    inference_client = get_bow_bert_emb_lc_parsect_infer(experiment_dirname)
