import json
import os
import sciwing.constants as constants
from sciwing.modules.embedders.bert_embedder import BertEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
import torch
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


def build_sectlabel_bow_bert(dirname: str):
    exp_dirpath = pathlib.Path(dirname)
    DATA_PATH = pathlib.Path(DATA_DIR)

    train_file = DATA_PATH.joinpath("sectLabel.train")
    dev_file = DATA_PATH.joinpath("sectLabel.dev")
    test_file = DATA_PATH.joinpath("sectLabel.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    embedder = BertEmbedder(
        dropout_value=0.0,
        aggregation_type="average",
        bert_type="bert-base-uncased",
        device=torch.device("cpu"),
    )

    encoder = BOW_Encoder(embedder=embedder, aggregation_type="average")
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=768,
        num_classes=23,
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    parsect_inference = ClassificationInference(
        model=model,
        model_filepath=str(exp_dirpath.joinpath("checkpoints", "best_model.pt")),
        datasets_manager=data_manager,
    )

    return parsect_inference


if __name__ == "__main__":
    dirname = pathlib.Path(".", "output")
    infer = build_sectlabel_bow_bert(str(dirname))
    infer.run_inference()
    infer.report_metrics()
