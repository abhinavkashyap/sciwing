import json
import os
import parsect.constants as constants
from parsect.modules.bow_bert_encoder import BowBertEncoder
from parsect.models.bow_bert_linear_classifier import BowBertLinearClassifier
from parsect.clients.parsect_inference import ParsectInference
import torch

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_bert_emb_bow_linear_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    BERT_TYPE = config["BERT_TYPE"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    DEVICE = config["DEVICE"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    encoder = BowBertEncoder(
        emb_dim=EMBEDDING_DIM,
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE)
    )

    model = BowBertLinearClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "bow_bert_base_cased_emb_lc_10e_1e-2lr"
    )
    inference_client = get_bert_emb_bow_linear_classifier_infer(experiment_dirname)
