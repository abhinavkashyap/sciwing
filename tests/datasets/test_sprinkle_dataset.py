import pytest
import sciwing.constants as constants
from sciwing.datasets.classification.sectlabel_dataset import SectLabelDataset
from sciwing.datasets.seq_labeling.parscit_dataset import ParscitDataset
import pathlib
from sciwing.utils.common import write_nfold_parscit_train_test


FILES = constants.FILES
PATHS = constants.PATHS
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture(scope="session")
def parsect_dataset(tmpdir_factory):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 10
    vocab_store_location = tmpdir_factory.mktemp("tempdir").join("vocab.json")
    DEBUG = True

    train_dataset = SectLabelDataset(
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


@pytest.fixture(scope="session")
def parscit_dataset(tmpdir_factory):
    parscit_train_filepath = pathlib.Path(PARSCIT_TRAIN_FILE)
    train_file = pathlib.Path(DATA_DIR, "parscit_train_conll.txt")
    test_file = pathlib.Path(DATA_DIR, "parscit_test_conll.txt")
    is_write_success = next(
        write_nfold_parscit_train_test(
            parscit_train_filepath,
            output_train_filepath=train_file,
            output_test_filepath=test_file,
        )
    )
    MAX_NUM_WORDS = 1000
    MAX_CHAR_LENGTH = 10
    MAX_INSTANCE_LENGTH = 10
    WORD_VOCAB_STORE_LOCATION = tmpdir_factory.mktemp("tempdir").join("vocab.json")
    CHAR_VOCAB_STORE_LOCATION = tmpdir_factory.mktemp("tempdir_char").join(
        "char_vocab.json"
    )
    CAPITALIZATION_VOCAB_STORE_LOCATION = None
    CAPITALIZATION_EMBEDDING_DIMENSION = None
    DEBUG = True
    DEBUG_DATASET_PROPORTION = 0.1
    WORD_EMBEDDING_TYPE = "random"
    WORD_EMBEDDING_DIMENSION = 100
    CHAR_EMBEDDING_DIMENSION = 10

    dataset = ParscitDataset(
        filename=str(train_file),
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_INSTANCE_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=WORD_VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        captialization_vocab_store_location=CAPITALIZATION_VOCAB_STORE_LOCATION,
        capitalization_emb_dim=CAPITALIZATION_EMBEDDING_DIMENSION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=WORD_EMBEDDING_TYPE,
        word_embedding_dimension=WORD_EMBEDDING_DIMENSION,
        char_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        word_start_token="<SOS>",
        word_end_token="<EOS>",
        word_pad_token="<PAD>",
        word_unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    options = {
        "MAX_NUM_WORDS": MAX_NUM_WORDS,
        "MAX_INSTANCE_LENGTH": MAX_INSTANCE_LENGTH,
        "MAX_CHAR_LENGTH": MAX_CHAR_LENGTH,
        "WORD_VOCAB_STORE_LOCATION": WORD_VOCAB_STORE_LOCATION,
        "CHAR_VOCAB_STORE_LOCATION": CHAR_VOCAB_STORE_LOCATION,
        "CAPITALIZATION_VOCAB_STORE_LOCATION": CAPITALIZATION_VOCAB_STORE_LOCATION,
        "CAPITALIZATION_EMBEDDING_DIMENSION": CAPITALIZATION_EMBEDDING_DIMENSION,
        "DEBUG": DEBUG,
        "DEBUG_DATASET_PROPORTION": DEBUG_DATASET_PROPORTION,
        "WORD_EMBEDDING_TYPE": WORD_EMBEDDING_TYPE,
        "WORD_EMBEDDING_DIMENSION": WORD_EMBEDDING_DIMENSION,
        "CHAR_EMBEDDING_DIMENSION": CHAR_EMBEDDING_DIMENSION,
    }

    return dataset, options


class TestSprinkleOnSectLabelDataset:
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


class TestSprinkleOnParscitDataset:
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
            "max_num_chars",
            "char_vocab_store_location",
            "char_embedding_type",
            "char_embedding_dimension",
            "char_unk_token",
            "char_pad_token",
            "char_start_token",
            "char_end_token",
        ],
    )
    def test_decorated_instance_has_attribute(self, attribute, parscit_dataset):
        dataset, options = parscit_dataset
        assert hasattr(dataset, attribute) and getattr(dataset, attribute) is not None
