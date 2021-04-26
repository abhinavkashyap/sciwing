import pytest
from sciwing.datasets.summarization.extractive_text_summarization_dataset import (
    ExtractiveSummarizationDataset,
    ExtractiveSummarizationDatasetManager,
)
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.utils.class_nursery import ClassNursery


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
        assert len(doc0) == len(label0.tokens["seq_label"])


@pytest.fixture(scope="session")
def ext_sum_dataset_manager(tmpdir_factory, request):
    train_file = tmpdir_factory.mktemp("train_data").join("train_file.txt")
    train_file.write(
        "trainword111 word112\tword121 word 122\tword 131###label11,label12,label13###refword12 refword22\n"
        "trainword211 word212\tword221###label21,label22###refword21"
    )

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev_file.txt")
    dev_file.write(
        "devword111 word112\tword121 word 122\tword 131###label11,label12,label13###refword12 refword22\n"
        "devword211 word212\tword221###label21,label22###refword21"
    )

    test_file = tmpdir_factory.mktemp("test_data").join("test_file.txt")
    test_file.write(
        "testword111 word112\tword121 word 122\tword 131###label11,label12,label13###refword12 refword22\n"
        "testword211 word212\tword221###label21,label22###refword21"
    )

    ext_sum_dataset_manager = ExtractiveSummarizationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return ext_sum_dataset_manager


class TestDatasetManager:
    def test_namespaces(self, ext_sum_dataset_manager):
        namespaces = ext_sum_dataset_manager.namespaces
        assert set(namespaces) == {"tokens", "char_tokens", "label"}

    def test_namespace_to_vocab(self, ext_sum_dataset_manager):
        namespace_to_vocab = ext_sum_dataset_manager.namespace_to_vocab
        assert namespace_to_vocab["tokens"].get_vocab_len() == 2 + 4
        # there is no special vocab here
        assert namespace_to_vocab["label"].get_vocab_len() == 2

    def test_namespace_to_numericalizers(self, ext_sum_dataset_manager):
        namespace_to_numericalizer = ext_sum_dataset_manager.namespace_to_numericalizer
        assert set(namespace_to_numericalizer.keys()) == {
            "tokens",
            "char_tokens",
            "label",
        }

    def test_label_namespace(self, ext_sum_dataset_manager):
        label_namespaces = ext_sum_dataset_manager.label_namespaces
        assert label_namespaces == ["label"]

    def test_num_labels(self, ext_sum_dataset_manager):
        num_labels = ext_sum_dataset_manager.num_labels["label"]
        assert num_labels == 2

    def test_print_stats(self, ext_sum_dataset_manager):
        try:
            ext_sum_dataset_manager.print_stats()
        except:
            pytest.fail(f"Print Stats fail to work in datasets manager")

    def test_texclassification_dataset_manager_in_nursery(
        self, ext_sum_dataset_manager
    ):
        assert (
            ClassNursery.class_nursery["TextClassificationDatasetManager"] is not None
        )
