import pytest
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)


@pytest.fixture(scope="session")
def clf_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train_file.txt")
    train_file.write("train_line1###label1\ntrain_line2###label2")

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev_file.txt")
    dev_file.write("dev_line1###label1\ndev_line2###label2")

    test_file = tmpdir_factory.mktemp("test_data").join("test_file.txt")
    test_file.write("dev_line1###label1\ndev_line2###label2")

    clf_dataset_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return clf_dataset_manager


class TestDatasetManager:
    def test_namespaces(self, clf_dataset_manager):
        namespaces = clf_dataset_manager.namespaces
        assert set(namespaces) == {
            "tokens",
            "label",
            "char_tokens",
            "bert_base_uncased_tokens",
            "bert_base_cased_tokens",
            "scibert_base_uncased_tokens",
            "scibert_base_cased_tokens",
        }

    def test_namespace_to_vocab(self, clf_dataset_manager):
        namespace_to_vocab = clf_dataset_manager.namespace_to_vocab
        assert namespace_to_vocab["tokens"].get_vocab_len() == 2 + 4
        assert namespace_to_vocab["label"].get_vocab_len() == 2 + 4

    def test_namespace_to_numericalizers(self, clf_dataset_manager):
        namespace_to_numericalizer = clf_dataset_manager.namespace_to_numericalizer
        assert set(namespace_to_numericalizer.keys()) == {
            "tokens",
            "label",
            "char_tokens",
            "bert_base_uncased_tokens",
            "bert_base_cased_tokens",
            "scibert_base_uncased_tokens",
            "scibert_base_cased_tokens",
        }
