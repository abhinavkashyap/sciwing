from sciwing.datasets.classification.generic_sect_dataset import GenericSectDataset
import pytest
import sciwing.constants as constants
from torch.utils.data import DataLoader
from sciwing.utils.class_nursery import ClassNursery

FILES = constants.FILES
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


@pytest.fixture(scope="session", params=["train", "valid", "test"])
def setup_generic_sect_dataset(tmpdir_factory, request):
    vocab_store_location = tmpdir_factory.mktemp("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 100
    MAX_LENGTH = 5
    DEBUG_DATASET_PROPORTION = 0.1
    EMBEDDING_TYPE = "random"
    EMBEDDING_DIMENSION = 300
    dataset_type = request.param

    dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type=dataset_type,
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
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
    def test_train_validation_test_split_numbers(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        headers = ["a"] * 100
        labels = [dataset.idx2classname[0]] * 100
        (train_headers, train_labels), (valid_headers, valid_labels), (
            test_headers,
            test_labels,
        ) = dataset.get_train_valid_test_stratified_split(
            headers, labels, dataset.classname2idx
        )

        assert len(train_headers) == 80
        assert len(train_labels) == 80
        assert len(valid_headers) == 10
        assert len(valid_labels) == 10
        assert len(test_headers) == 10
        assert len(test_labels) == 10

    def test_num_labels(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        classname2idx = dataset.classname2idx
        num_labels = len(list(classname2idx.keys()))
        assert num_labels == 12

    def test_no_train_header_empty(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        headers, labels = dataset.get_lines_labels(GENERIC_SECTION_TRAIN_FILE)
        assert all([bool(header.strip()) for header in headers])

    def test_no_train_label_empty(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        headers, labels = dataset.get_lines_labels(filename=GENERIC_SECTION_TRAIN_FILE)
        assert all([bool(label.strip()) for label in labels])

    def test_embedding_has_values(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        embedding_tensors = dataset.word_vocab.load_embedding()
        assert embedding_tensors.size(0) > 0
        assert embedding_tensors.size(1) == options["EMBEDDING_DIMENSION"]

    def test_train_loader(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        loader = DataLoader(dataset=dataset, batch_size=3)
        instances_dict = next(iter(loader))
        assert len(instances_dict["instance"]) == 3
        assert instances_dict["tokens"].size(0) == 3
        assert instances_dict["tokens"].size(1) == options["MAX_LENGTH"]
        assert instances_dict["label"].size(0) == 3

    def test_iter_dict_should_have_tokens_labels(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        iter_dict = dataset[0]
        assert "label" in iter_dict.keys()
        assert "tokens" in iter_dict.keys()

    def test_get_stats_works(self, setup_generic_sect_dataset):
        dataset, options = setup_generic_sect_dataset
        try:
            dataset.print_stats()
        except:
            pytest.fail("print_stats() of GenericSect Dataset fails")

    def test_generic_sect_in_nursery(self):
        assert ClassNursery.class_nursery.get("GenericSectDataset") is not None
