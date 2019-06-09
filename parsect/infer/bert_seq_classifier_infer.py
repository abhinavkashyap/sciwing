import parsect.constants as constants
import json
from parsect.models.bert_seq_classifier import BertSeqClassifier
from parsect.clients.parsect_inference import ParsectInference
import os

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_bert_seq_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    NUM_CLASSES = config["NUM_CLASSES"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    BERT_TYPE = config["BERT_TYPE"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    model = BertSeqClassifier(
        num_classes=NUM_CLASSES,
        emb_dim=EMBEDDING_DIMENSION,
        bert_type=BERT_TYPE,
        dropout_value=0.0
    )

    inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath
    )

    return inference


if __name__ == "__main__":
    dirname = os.path.join(OUTPUT_DIR, "debug_bert_seq_classifier_base_cased_emb_lc_10e_1e-2lr")
    infer = get_bert_seq_classifier_infer(dirname)
