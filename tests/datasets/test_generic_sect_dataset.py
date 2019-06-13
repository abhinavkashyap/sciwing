from parsect.datasets.generic_sect_dataset import GenericSectDataset
import pytest
import parsect.constants as constants
from torch.utils.data import DataLoader


FILES = constants.FILES
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


@pytest.fixture
def setup_generic_sect_train_dataset(tmpdir):
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 100
    MAX_LENGTH = 5
    DEBUG_DATASET_PROPORTION = 0.1
    EMBEDDING_TYPE = "random"
    EMBEDDING_DIMENSION = 300

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    return (
        dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "DEBUG": DEBUG,
            "DEBUG_DATASET_PROPORTION": DEBUG_DATASET_PROPORTION,
            "EMBEDDING_TYPE": EMBEDDING_TYPE,
            "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        },
    )


@pytest.fixture
def setup_generic_sect_valid_dataset(tmpdir):
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 100
    MAX_LENGTH = 5
    DEBUG_DATASET_PROPORTION = 0.1
    EMBEDDING_TYPE = "random"
    EMBEDDING_DIMENSION = 300

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    return (
        dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "DEBUG": DEBUG,
            "DEBUG_DATASET_PROPORTION": DEBUG_DATASET_PROPORTION,
            "EMBEDDING_TYPE": EMBEDDING_TYPE,
            "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        },
    )


@pytest.fixture
def setup_generic_sect_test_dataset(tmpdir):
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 100
    MAX_LENGTH = 5
    DEBUG_DATASET_PROPORTION = 0.1
    EMBEDDING_TYPE = "random"
    EMBEDDING_DIMENSION = 300

    dataset = GenericSectDataset(
        generic_sect_filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    return (
        dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "DEBUG": DEBUG,
            "DEBUG_DATASET_PROPORTION": DEBUG_DATASET_PROPORTION,
            "EMBEDDING_TYPE": EMBEDDING_TYPE,
            "EMBEDDING_DIMENSION": EMBEDDING_DIMENSION,
        },
    )


class TestGenericSectDataset:
    def test_train_validation_test_split_numbers(
        self, setup_generic_sect_train_dataset
    ):
        dataset, options = setup_generic_sect_train_dataset
        headers = ["a"] * 100
        labels = [dataset.idx2label[0]] * 100
        (train_headers, train_labels), (valid_headers, valid_labels), (
            test_headers,
            test_labels,
        ) = dataset.get_train_valid_test_split(headers, labels)

        assert len(train_headers) == 80
        assert len(train_labels) == 80
        assert len(valid_headers) == 10
        assert len(valid_labels) == 10
        assert len(test_headers) == 10
        assert len(test_labels) == 10

    def test_num_labels(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        label2idx = dataset.label2idx
        num_labels = len(list(label2idx.keys()))
        assert num_labels == 12

    def test_no_train_header_empty(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(header.strip()) for header in headers])

    def test_no_train_label_empty(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(label.strip()) for label in labels])

    def test_no_valid_header_empty(self, setup_generic_sect_valid_dataset):
        dataset, options = setup_generic_sect_valid_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(header.strip()) for header in headers])

    def test_no_valid_label_empty(self, setup_generic_sect_valid_dataset):
        dataset, options = setup_generic_sect_valid_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(label.strip()) for label in labels])

    def test_no_test_header_empty(self, setup_generic_sect_test_dataset):
        dataset, options = setup_generic_sect_test_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(header.strip()) for header in headers])

    def test_no_test_label_empty(self, setup_generic_sect_test_dataset):
        dataset, options = setup_generic_sect_test_dataset
        headers, labels = dataset.get_header_labels()
        assert all([bool(label.strip()) for label in labels])

    def test_embedding_has_values(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        embedding_tensors = dataset.get_preloaded_embedding()
        assert embedding_tensors.size(0) > 0
        assert embedding_tensors.size(1) == options["EMBEDDING_DIMENSION"]

    def test_train_loader(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        loader = DataLoader(dataset=dataset, batch_size=3)
        instances_dict = next(iter(loader))
        assert len(instances_dict["instance"]) == 3
        assert instances_dict["tokens"].size(0) == 3
        assert instances_dict["tokens"].size(1) == options["MAX_LENGTH"]
        assert instances_dict["label"].size(0) == 3

    def test_iter_dict_should_have_tokens_labels(
        self, setup_generic_sect_train_dataset
    ):
        dataset, options = setup_generic_sect_train_dataset
        iter_dict = dataset[0]
        assert "label" in iter_dict.keys()
        assert "tokens" in iter_dict.keys()

    def test_get_stats_works(self, setup_generic_sect_train_dataset):
        dataset, options = setup_generic_sect_train_dataset
        try:
            dataset.get_stats()
        except:
            pytest.fail("get_stats() of GenericSect Dataset fails")
