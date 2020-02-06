import pytest
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDataset,
)
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write("line1###label1\nline2###label2")
    return p


class TestTextClassificationDataset:
    def test_get_lines_labels(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        lines = classification_dataset.lines
        assert len(lines) == 2

    def test_len_dataset(self, test_file):
        classification_dataset = TextClassificationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert len(classification_dataset) == 2

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
