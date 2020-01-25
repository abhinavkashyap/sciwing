import pytest
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDataset,
)
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write("line1###label1\nline2###label2")
    return p


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
        batch_size=2,
    )

    return clf_dataset_manager


class TestTextClassificationDataset:
    def test_get_lines_labels(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        lines = classification_dataset.lines
        assert len(lines) == 2

    def test_get_classname2idx(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        classname2idx = classification_dataset.get_classname2idx()
        assert len(classname2idx) == 2
        assert set(classname2idx.keys()) == {"label1", "label2"}

    def test_len_dataset(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert len(classification_dataset) == 2

    def test_num_classes(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert classification_dataset.get_num_classes() == 2

    def test_get_item(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        num_instances = len(classification_dataset)
        tokens = ["line1", "line2"]
        line_tokens = []
        for idx in range(num_instances):
            line, label = classification_dataset[idx]
            line_tokens.extend(line.tokens["tokens"])

        line_tokens = list(map(lambda token: token.text, line_tokens))

        assert set(tokens) == set(line_tokens)


class TestTextClassificationDatasetManager:
    @pytest.mark.parametrize("dataset_type", ["train", "dev", "test"])
    def test_iter_dict_namespaces(self, clf_dataset_manager, dataset_type):
        iter_dict = clf_dataset_manager._get_iter_dict(for_dataset=dataset_type)
        assert set(list(iter_dict.keys())) == {
            "tokens",
            "label",
            "char_tokens",
            "bert_base_uncased_tokens",
            "bert_base_cased_tokens",
            "scibert_base_uncased_tokens",
            "scibert_base_cased_tokens",
        }
        assert len(iter_dict["tokens"]) == 2
        assert len(iter_dict["label"]) == 2
