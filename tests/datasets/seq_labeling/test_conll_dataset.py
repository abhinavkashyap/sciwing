import pytest
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDataset
from sciwing.tokenizers.word_tokenizer import WordTokenizer


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
