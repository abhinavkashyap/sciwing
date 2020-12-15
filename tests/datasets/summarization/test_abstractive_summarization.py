import pytest
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import (
    AbstractiveSummarizationDataset
)
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write(
        "word11_train word21_train###word11_label word12_label\nword12_train word22_train word32_train###word21_label"
    )
    return p


class TestAbstractSummarizationDataset:
    def test_get_lines_labels(self, test_file):
        summarization_dataset = AbstractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        lines = summarization_dataset.lines
        assert len(lines) == 2

    def test_len_dataset(self, test_file):
        summarization_dataset = AbstractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert len(summarization_dataset) == 2

    def test_get_item(self, test_file):
        summarization_dataset = AbstractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        num_instances = len(summarization_dataset)
        defined_line_tokens = ["word11_train", "word21_train", "word12_train", "word22_train", "word32_train"]
        defined_label_tokens = ["word11_label", "word12_label", "word21_label"]
        line_tokens = []
        label_tokens = []
        for idx in range(num_instances):
            line, label = summarization_dataset[idx]
            line_tokens.extend(line.tokens["tokens"])
            label_tokens.extend(label.tokens["tokens"])

        line_tokens = list(map(lambda token: token.text, line_tokens))
        label_tokens = list(map(lambda token: token.text, label_tokens))

        assert set(defined_line_tokens) == set(line_tokens)
        assert set(defined_label_tokens) == set(label_tokens)
