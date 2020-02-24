import pytest
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.utils.class_nursery import ClassNursery


@pytest.fixture(scope="session")
def clf_dataset_manager(tmpdir_factory):
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


@pytest.fixture
def setup_simple_classifier(clf_dataset_manager):
    datasets_manager = clf_dataset_manager
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    encoder = BOW_Encoder(embedder=embedder)
    classifier = SimpleClassifier(
        encoder=encoder,
        encoding_dim=50,
        num_classes=2,
        datasets_manager=datasets_manager,
        classification_layer_bias=True,
    )
    train_dataset = datasets_manager.train_dataset
    lines, labels = train_dataset.get_lines_labels()

    return classifier, lines, labels


class TestSimpleClassifier:
    def test_classifier_logits_shape(self, setup_simple_classifier):
        classifier, lines, labels = setup_simple_classifier
        output = classifier(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )
        logits = output["logits"]
        assert logits.size(0) == 2
        assert logits.size(1) == 2

    def test_classifier_normalized_probs_shape(self, setup_simple_classifier):
        classifier, lines, labels = setup_simple_classifier
        output = classifier(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )
        normalized_probs = output["normalized_probs"]
        assert normalized_probs.size(0) == 2 and normalized_probs.size(1) == 2

    # def test_classifier_produces_equal_probs_for_0_embedding(
    #     self, setup_simple_classifier
    # ):
    #     iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
    #     output = simple_classifier(
    #         iter_dict, is_training=True, is_validation=False, is_test=False
    #     )
    #     probs = output["normalized_probs"]
    #     expected_probs = torch.ones([batch_size, num_classes]) / num_classes
    #     assert torch.allclose(probs, expected_probs)
    #
    # def test_classifier_produces_correct_initial_loss_for_0_embedding(
    #     self, setup_simple_classifier
    # ):
    #     iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
    #     output = simple_classifier(
    #         iter_dict, is_training=True, is_validation=False, is_test=False
    #     )
    #     loss = output["loss"].item()
    #     correct_loss = -np.log(1 / num_classes)
    #     assert torch.allclose(torch.Tensor([loss]), torch.Tensor([correct_loss]))

    # def test_classifier_produces_correct_precision(self, setup_simple_classifier):
    #     iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
    #     output = simple_classifier(
    #         iter_dict, is_training=True, is_validation=False, is_test=False
    #     )
    #     idx2labelname_mapping = {0: "good class", 1: "bad class", 2: "average_class"}
    #     metrics_calc = PrecisionRecallFMeasure(
    #         idx2labelname_mapping=idx2labelname_mapping
    #     )
    #
    #     metrics_calc.calc_metric(iter_dict=iter_dict, model_forward_dict=output)
    #     metrics = metrics_calc.get_metric()
    #     precision = metrics["precision"]
    #
    #     # NOTE: topk returns the last value in the dimension incase
    #     # all the values are equal.
    #     expected_precision = {1: 0, 2: 0}
    #
    #     assert len(precision) == 2
    #
    #     for class_label, precision_value in precision.items():
    #         assert precision_value == expected_precision[class_label]

    def test_simple_classifier_in_class_nursery(self):
        assert ClassNursery.class_nursery.get("SimpleClassifier") is not None
