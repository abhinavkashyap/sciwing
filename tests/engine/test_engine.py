from sciwing.engine.engine import Engine
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import torch
import os
from sciwing.utils.class_nursery import ClassNursery

import pytest
import sciwing.constants as constants

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


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
def setup_engine_test_with_simple_classifier(
    request, clf_datasets_manager, tmpdir_factory
):
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

    return engine


class TestEngine:
    def test_train_loader(self, setup_engine_test_with_simple_classifier):
        engine = setup_engine_test_with_simple_classifier
        train_dataset = engine.get_train_dataset()
        train_loader = engine.get_loader(train_dataset)

        for lines_labels in train_loader:
            for line, label in lines_labels:
                assert isinstance(line, Line)
                assert isinstance(label, Label)

    def test_one_train_epoch(self, setup_engine_test_with_simple_classifier):
        # check whether you can run train_epoch without throwing an error
        engine = setup_engine_test_with_simple_classifier
        engine.train_epoch(0)

    def test_save_model(self, setup_engine_test_with_simple_classifier):
        engine = setup_engine_test_with_simple_classifier
        engine.train_epoch_end(0)

        # test for the file model_epoch_1.pt
        assert os.path.isdir(engine.save_dir)
        assert os.path.isfile(os.path.join(engine.save_dir, "model_epoch_1.pt"))

    def test_runs(self, setup_engine_test_with_simple_classifier):
        """
        Just tests runs without any errors
        """
        engine = setup_engine_test_with_simple_classifier
        try:
            engine.run()
        except:
            pytest.fail("Engine failed to run")

    def test_load_model(self, setup_engine_test_with_simple_classifier):
        """
        Test whether engine loads the model without any error.
        """
        engine = setup_engine_test_with_simple_classifier
        try:
            engine.train_epoch_end(0)
            engine.load_model_from_file(
                os.path.join(engine.save_dir, "model_epoch_{0}.pt".format(1))
            )
        except:
            pytest.fail("Engine train epoch end failed")

    def test_engine_in_class_nursery(self):
        assert ClassNursery.class_nursery["Engine"] is not None
