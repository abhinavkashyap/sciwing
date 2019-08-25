import pytest
import torch
import torch.nn as nn
from parsect.modules.embedders import VanillaEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
import parsect.constants as constants
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
from parsect.datasets.classification.parsect_dataset import ParsectDataset
import pathlib
import json

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]


@pytest.fixture
def setup_inference(tmpdir_factory):
    debug_experiment_path = pathlib.Path(OUTPUT_DIR, "parsect_bow_random_emb_lc_debug")
    config_file = debug_experiment_path.joinpath(debug_experiment_path, "config.json")
    test_dataset_params = debug_experiment_path.joinpath("test_dataset_params.json")
    model_filepath = debug_experiment_path.joinpath("checkpoints", "best_model.pt")

    with open(config_file) as fp:
        config = json.load(fp)

    EMB_DIM = config.get("EMBEDDING_DIMENSION")
    VOCAB_SIZE = config.get("VOCAB_SIZE")
    NUM_CLASSES = config.get("NUM_CLASSES")

    # setup the model
    embedding = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    embedder = VanillaEmbedder(embedding_dim=EMB_DIM, embedding=embedding)

    encoder = BOW_Encoder(
        emb_dim=EMB_DIM, embedder=embedder, dropout_value=0, aggregation_type="sum"
    )

    simple_classifier = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )
    # setup the dataset
    with open(test_dataset_params) as fp:
        test_dataset_params = json.load(fp)

    dataset = ParsectDataset(**test_dataset_params)

    inference = ClassificationInference(
        model=simple_classifier, model_filepath=model_filepath, dataset=dataset
    )

    return inference, dataset


class TestClassificationInference:
    def test_run_inference_works(self, setup_inference):
        inference_client, dataset = setup_inference
        try:
            inference_client.run_inference()
        except:
            pytest.fail("Run inference for classification dataset fails")

    def test_run_test_works(self, setup_inference):
        inference_client, dataset = setup_inference
        try:
            inference_client.run_inference()
        except:
            pytest.fail("Run test doest not work")

    def test_on_user_input_works(self, setup_inference):
        inference_client, dataset = setup_inference
        try:
            inference_client.on_user_input(line="test input")
        except:
            pytest.fail("On user input fails")

    def test_print_metrics_works(self, setup_inference):
        inference_client, dataset = setup_inference
        inference_client.run_test()
        try:
            inference_client.print_metrics()
        except:
            pytest.fail("Print metrics failed")

    def test_print_confusion_metrics_works(self, setup_inference):
        inference_client, dataset = setup_inference
        inference_client.run_test()
        try:
            inference_client.print_confusion_matrix()
        except:
            pytest.fail("Print confusion matrix fails")

    def test_get_misclassified_sentences(self, setup_inference):
        inference_client, dataset = setup_inference
        inference_client.run_test()
        try:
            inference_client.get_misclassified_sentences(
                true_label_idx=0, pred_label_idx=1
            )
        except:
            pytest.fail("Getting misclassified sentence fail")
