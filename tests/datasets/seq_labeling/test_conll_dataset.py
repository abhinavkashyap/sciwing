import pytest
from sciwing.datasets.seq_labeling.conll_dataset import (
    CoNLLDataset,
    CoNLLDatasetManager,
)
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.preprocessing.instance_preprocessing import InstancePreprocessing


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write("word1 O-Task O-Process O-Material\nword2 B-Task B-Process O-Material")
    return p


class TestCoNLLDataset:
    def test_get_lines_labels_len(self, test_file):
        dataset = CoNLLDataset(
            filename=test_file, tokenizers={"tokens": WordTokenizer()}
        )

        lines, labels = dataset.get_lines_labels()
        assert len(lines) == 1
        assert len(labels) == 1

    def test_labels_namespaces(self, test_file):
        dataset = CoNLLDataset(
            filename=test_file,
            tokenizers={"tokens": WordTokenizer()},
            column_names=["NER", "POS", "DEP"],
        )
        lines, labels = dataset.get_lines_labels()
        for label in labels:
            namespaces = label.namespace
            assert len(namespaces) == 3
            assert "NER" in namespaces
            assert "POS" in namespaces
            assert "DEP" in namespaces

    def test_len_lines_labels_equal(self, test_file):
        dataset = CoNLLDataset(
            filename=test_file,
            tokenizers={"tokens": WordTokenizer()},
            column_names=["NER", "POS", "DEP"],
        )
        lines, labels = dataset.get_lines_labels()
        for line, label in zip(lines, labels):
            line_tokens = line.tokens["tokens"]
            labels_ner = label.tokens["NER"]
            labels_pos = label.tokens["POS"]
            labels_dep = label.tokens["DEP"]
            assert (
                len(line_tokens)
                == len(labels_ner)
                == len(labels_pos)
                == len(labels_dep)
            )

    @pytest.mark.parametrize("train_only", ["ner", "pos", "dep"])
    def test_restricted_namesapces(self, test_file, train_only):
        dataset = CoNLLDataset(
            filename=test_file,
            tokenizers={"tokens": WordTokenizer()},
            column_names=["POS", "DEP", "NER"],
            train_only=train_only,
        )
        lines, labels = dataset.get_lines_labels()

        for label in labels:
            namespaces = label.namespace
            assert len(namespaces) == 1
            assert train_only.upper() in namespaces

    def test_conll_dataset_manager(self, test_file):
        instance_preprocessing = InstancePreprocessing()
        manager = CoNLLDatasetManager(
            train_filename=test_file,
            dev_filename=test_file,
            test_filename=test_file,
            namespace_vocab_options={
                "tokens": {
                    "preprocessing_pipeline": [instance_preprocessing.lowercase],
                    "include_special_vocab": False,
                }
            },
        )

        token_vocab = manager.namespace_to_vocab["tokens"].get_token2idx_mapping()

        for token in token_vocab.keys():
            assert token.islower()
