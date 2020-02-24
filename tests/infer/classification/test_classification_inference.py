import pytest
import sciwing.constants as constants
import pathlib
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.engine.engine import Engine
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import torch
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]


@pytest.fixture(scope="session")
def clf_datasets_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train_file.txt")
    train_file.write("train_line1###label1\ntrain_line2###label2")

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev_file.txt")
    dev_file.write("dev_line1###label1\ndev_line2###label2")

    test_file = tmpdir_factory.mktemp("test_data").join("test_file.txt")
    test_file.write("test_line1###label1\ntest_line2###label2")

    clf_dataset_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
        batch_size=1,
    )

    return clf_dataset_manager


@pytest.fixture(scope="session", params=["loss", "micro_fscore", "macro_fscore"])
def setup_sectlabel_bow_glove_infer(request, clf_datasets_manager, tmpdir_factory):
    track_for_best = request.param
    sample_proportion = 0.5
    datasets_manager = clf_datasets_manager
    word_embedder = WordEmbedder(embedding_type="glove_6B_50")
    bow_encoder = BOW_Encoder(embedder=word_embedder)
    classifier = SimpleClassifier(
        encoder=bow_encoder,
        encoding_dim=word_embedder.get_embedding_dimension(),
        num_classes=2,
        classification_layer_bias=True,
        datasets_manager=datasets_manager,
    )
    train_metric = PrecisionRecallFMeasure(datasets_manager=datasets_manager)
    validation_metric = PrecisionRecallFMeasure(datasets_manager=datasets_manager)
    test_metric = PrecisionRecallFMeasure(datasets_manager=datasets_manager)

    optimizer = torch.optim.Adam(params=classifier.parameters())
    batch_size = 1
    save_dir = tmpdir_factory.mktemp("experiment_1")
    num_epochs = 1
    save_every = 1
    log_train_metrics_every = 10

    engine = Engine(
        model=classifier,
        datasets_manager=datasets_manager,
        optimizer=optimizer,
        batch_size=batch_size,
        save_dir=save_dir,
        num_epochs=num_epochs,
        save_every=save_every,
        log_train_metrics_every=log_train_metrics_every,
        train_metric=train_metric,
        validation_metric=validation_metric,
        test_metric=test_metric,
        track_for_best=track_for_best,
        sample_proportion=sample_proportion,
    )

    engine.run()
    model_filepath = pathlib.Path(save_dir).joinpath("best_model.pt")
    infer = ClassificationInference(
        model=classifier,
        model_filepath=str(model_filepath),
        datasets_manager=datasets_manager,
    )
    return infer


class TestClassificationInference:
    def test_run_inference_works(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        try:
            inference_client.run_inference()
        except:
            pytest.fail("Run inference for classification dataset fails")

    def test_run_test_works(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        try:
            inference_client.run_test()
        except:
            pytest.fail("Run test doest not work")

    def test_on_user_input_works(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        try:
            inference_client.on_user_input(line="test input")
        except:
            pytest.fail("On user input fails")

    def test_print_metrics_works(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        inference_client.run_test()
        try:
            inference_client.report_metrics()
        except:
            pytest.fail("Print metrics failed")

    def test_print_confusion_metrics_works(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        inference_client.run_test()
        try:
            inference_client.print_confusion_matrix()
        except:
            pytest.fail("Print confusion matrix fails")

    def test_get_misclassified_sentences(self, setup_sectlabel_bow_glove_infer):
        inference_client = setup_sectlabel_bow_glove_infer
        inference_client.run_test()
        try:
            inference_client.get_misclassified_sentences(
                true_label_idx=0, pred_label_idx=1
            )
        except:
            pytest.fail("Getting misclassified sentence fail")
