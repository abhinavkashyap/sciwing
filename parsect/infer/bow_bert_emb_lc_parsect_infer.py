import json
import os
import parsect.constants as constants
from parsect.modules.embedders.bert_embedder import BertEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.classification.sectlabel_dataset import SectLabelDataset
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
import torch
import pathlib

PATHS = constants.PATHS
FILES = constants.FILES
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_bow_bert_emb_lc_parsect_infer(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    hyperparam_config_filepath = exp_dirpath.joinpath("config.json")
    test_dataset_params = exp_dirpath.joinpath("test_dataset_params.json")

    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    with open(test_dataset_params, "r") as fp:
        test_dataset_args = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    BERT_TYPE = config["BERT_TYPE"]

    DEVICE = config["DEVICE"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    embedder = BertEmbedder(
        emb_dim=EMBEDDING_DIM,
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE),
    )

    encoder = BOW_Encoder(
        embedder=embedder, emb_dim=EMBEDDING_DIM, aggregation_type="average"
    )
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    dataset = SectLabelDataset(**test_dataset_args)

    parsect_inference = ClassificationInference(
        model=model, model_filepath=model_filepath, dataset=dataset
    )

    return parsect_inference


if __name__ == "__main__":
    experiment_dirname = os.path.join(
        OUTPUT_DIR, "debug_bow_bert_base_cased_emb_lc_10e_1e-2lr"
    )
    inference_client = get_bow_bert_emb_lc_parsect_infer(experiment_dirname)
