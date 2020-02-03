import pytest
from sciwing.datasets.seq_labeling.seq_labelling_dataset import SeqLabellingDataset
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write(
        "word11###label1 word21###label2\nword12###label1 word22###label2 word32###label3"
    )
    return p


class TestSeqLabellingDataset:
    def test_get_lines_labels(self, test_file):
        dataset = SeqLabellingDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        lines, labels = dataset.get_lines_labels()
        assert len(lines) == 2

    def test_len(self, test_file):
        dataset = SeqLabellingDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert len(dataset) == 2

    def test_get_item(self, test_file):
        dataset = SeqLabellingDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        num_instances = len(dataset)

        for idx in range(num_instances):
            line, label = dataset[idx]
            word_tokens = line.tokens["tokens"]
            label_tokens = label.tokens["seq_label"]
            print(f"label tokens {label.tokens}")
            assert len(word_tokens) == len(label_tokens)
