import pytest
from sciwing.datasets.seq_labeling.conll_yago_dataset import ConllYagoDatasetsManager
from sciwing.datasets.seq_labeling.conll_yago_dataset import ConllYagoDataset
import sciwing.constants as constants
import pathlib
from sciwing.tokenizers.word_tokenizer import WordTokenizer

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


@pytest.fixture(
    params=["conll_yago_ner.train", "conll_yago_ner.dev", "conll_yago_ner.test"]
)
def conll_yago_dataset(request):
    train_filename = DATA_DIR.joinpath(request.param)
    dataset = ConllYagoDataset(
        filename=str(train_filename),
        tokenizers={"tokens": WordTokenizer(tokenizer="vanilla")},
        column_names=["NER"],
    )

    return dataset


@pytest.fixture()
def conll_yago_dataset_manager():
    train_filename = DATA_DIR.joinpath("conll_yago_ner.train")
    dev_filename = DATA_DIR.joinpath("conll_yago_ner.dev")
    test_filename = DATA_DIR.joinpath("conll_yago_ner.test")

    dataset_manager = ConllYagoDatasetsManager(
        train_filename=str(train_filename),
        dev_filename=str(dev_filename),
        test_filename=str(test_filename),
    )

    return dataset_manager


class TestConllYagoDataset:
    def test_get_lines_labels(self, conll_yago_dataset):
        dataset = conll_yago_dataset
        try:
            lines, labels = dataset.get_lines_labels()
            assert len(lines) > 0
            assert len(labels) > 0
        except:
            pytest.fail("Getting Lines and Labels failed")

    def test_labels_namespace(self, conll_yago_dataset):
        dataset = conll_yago_dataset
        lines, labels = dataset.get_lines_labels()
        for label in labels:
            namespaces = label.namespace
            assert len(namespaces) == 1
            assert "NER" in namespaces

    def test_lines_labels_length(self, conll_yago_dataset):
        dataset = conll_yago_dataset
        lines, labels = dataset.get_lines_labels()
        for line, label in zip(lines, labels):
            line_tokens = line.tokens["tokens"]
            labels_ner = label.tokens["NER"]
            assert len(line_tokens) == len(labels_ner)

    def test_conll_yago_dataset_manager(self, conll_yago_dataset_manager):
        dataset_manager = conll_yago_dataset_manager
        tokens_vocab = dataset_manager.namespace_to_vocab["tokens"]
        assert tokens_vocab.get_vocab_len() > 0

    def test_context_lines_tokens_namespace(self, conll_yago_dataset):
        dataset = conll_yago_dataset
        lines, labels = dataset.get_lines_labels()
        for line in lines:
            tokens = line.context_tokens
            assert "tokens" in tokens.keys()
