import json
import os
import parsect.constants as constants
from parsect.clients.parsect_inference import ParsectInference
from parsect.models.bow_elmo_linear_classifier import BowElmoLinearClassifier
from parsect.modules.bow_elmo_encoder import BowElmoEncoder

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]


def get_elmo_emb_linear_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    encoder = BowElmoEncoder(emb_dim=EMBEDDING_DIM)

    NUM_CLASSES = config["NUM_CLASSES"]

    model = BowElmoLinearClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES
    )

    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    parsect_inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath
    )

    return parsect_inference


if __name__ == '__main__':
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "debug_bow_elmo_emb_lc_50e_1e-4lr"
    )
    inference_client = get_elmo_emb_linear_classifier_infer(experiment_dirname)
