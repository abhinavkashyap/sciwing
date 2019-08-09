import pytest
import parsect.constants as constants
from parsect.datasets.classification.parsect_dataset import ParsectDataset

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


@pytest.fixture(scope="session")
def parsect_dataset(tmpdir_factory):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 10
    vocab_store_location = tmpdir_factory.mktemp("tempdir").join("vocab.json")
    DEBUG = True

    train_dataset = ParsectDataset(
        filename=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        train_size=0.8,
        test_size=0.2,
        validation_size=0.5,
    )

    return (
        train_dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "vocab_store_location": vocab_store_location,
        },
    )


class TestSprinklClfDataset:
    @pytest.mark.parametrize(
        "attribute",
        [
            "filename",
            "dataset_type",
            "max_num_words",
            "max_instance_length",
            "word_vocab_store_location",
            "debug",
            "debug_dataset_proportion",
            "word_embedding_type",
            "word_embedding_dimension",
            "word_start_token",
            "word_end_token",
            "word_pad_token",
            "word_unk_token",
            "train_size",
            "test_size",
            "validation_size",
            "word_tokenizer",
            "word_tokenization_type",
            "word_vocab",
            "num_instances",
            "label_stats_table",
        ],
    )
    def test_decorated_instance_has_attribute(self, attribute, parsect_dataset):
        dataset, options = parsect_dataset
        assert hasattr(dataset, attribute)

    @pytest.mark.parametrize(
        "attribute, exp_value",
        [
            ("filename", SECT_LABEL_FILE),
            ("dataset_type", "train"),
            ("max_num_words", 1000),
            ("max_instance_length", 10),
            ("train_size", 0.8),
            ("test_size", 0.2),
            ("validation_size", 0.5),
        ],
    )
    def test_decorated_instance_has_correct_values_for(
        self, parsect_dataset, attribute, exp_value
    ):
        dataset, options = parsect_dataset
        value = getattr(dataset, attribute)
        assert value == exp_value
