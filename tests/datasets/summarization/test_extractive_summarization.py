import pytest
from sciwing.datasets.summarization.extractive_text_summarization_dataset import ExtractiveSummarizationDataset
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture(scope="session")
def test_file(tmpdir_factory):
    p = tmpdir_factory.mktemp("data").join("test.txt")
    p.write(
        "word111 word112\tword121 word 122\tword 131###label11,label12,label13###refword12 refword22\n"
        "word211 word212\tword221###label21,label22###refword21"
    )
    return p


class TestExtractiveSummarizationDataset:
    def test_get_lines_labels_refs(self, test_file):
        dataset = ExtractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        doc, labels, ref = dataset.get_docs_labels_refs()
        assert len(doc) == 2

    def test_len(self, test_file):
        dataset = ExtractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )
        assert len(dataset) == 2

    def test_get_item(self, test_file):
        dataset = ExtractiveSummarizationDataset(
            filename=str(test_file), tokenizers={"tokens": WordTokenizer()}
        )

        doc0, label0, ref0 = dataset[0]
        assert len(doc0) == len(label0.tokens['seq_label'])

