import parsect.constants as constants
import pytest
from parsect.utils.common import write_nfold_parscit_train_test
from parsect.datasets.parscit_dataset import ParscitDataset
from torch.utils.data import DataLoader

import pathlib

FILES = constants.FILES
PATHS = constants.PATHS
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture()
def setup_parscit_train_dataset(tmpdir):
    parscit_train_filepath = pathlib.Path(PARSCIT_TRAIN_FILE)
    is_write_success = next(write_nfold_parscit_train_test(parscit_train_filepath))
    train_file = pathlib.Path(DATA_DIR, "parscit_train_conll.txt")
    test_file = pathlib.Path(DATA_DIR, "parscit_test_conll.txt")
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 10000
    MAX_LENGTH = 20
    EMBEDDING_DIM = 100
    train_dataset = None
    test_dataset = None

    if is_write_success:
        train_dataset = ParscitDataset(
            parscit_conll_file=str(train_file),
            dataset_type="train",
            max_num_words=MAX_NUM_WORDS,
            max_length=MAX_LENGTH,
            vocab_store_location=vocab_store_location,
            debug=DEBUG,
            embedding_type="random",
            embedding_dimension=EMBEDDING_DIM,
            add_start_end_token=False,
        )
        test_dataset = ParscitDataset(
            parscit_conll_file=str(test_file),
            dataset_type="train",
            max_num_words=MAX_NUM_WORDS,
            max_length=MAX_LENGTH,
            vocab_store_location=vocab_store_location,
            debug=DEBUG,
            embedding_type="random",
            embedding_dimension=EMBEDDING_DIM,
            add_start_end_token=False,
        )

    options = {
        "MAX_NUM_WORDS": MAX_NUM_WORDS,
        "MAX_LENGTH": MAX_LENGTH,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }

    return train_dataset, test_dataset, options


@pytest.fixture()
def setup_parscit_train_dataset_maxlen_2(tmpdir):
    parscit_train_filepath = pathlib.Path(PARSCIT_TRAIN_FILE)
    is_write_success = next(write_nfold_parscit_train_test(parscit_train_filepath))
    train_file = pathlib.Path(DATA_DIR, "parscit_train_conll.txt")
    test_file = pathlib.Path(DATA_DIR, "parscit_test_conll.txt")
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 10000
    MAX_LENGTH = 2
    EMBEDDING_DIM = 100
    train_dataset = None
    test_dataset = None

    if is_write_success:
        train_dataset = ParscitDataset(
            parscit_conll_file=str(train_file),
            dataset_type="train",
            max_num_words=MAX_NUM_WORDS,
            max_length=MAX_LENGTH,
            vocab_store_location=vocab_store_location,
            debug=DEBUG,
            embedding_type="random",
            embedding_dimension=EMBEDDING_DIM,
            add_start_end_token=True,
        )
        test_dataset = ParscitDataset(
            parscit_conll_file=str(test_file),
            dataset_type="train",
            max_num_words=MAX_NUM_WORDS,
            max_length=MAX_LENGTH,
            vocab_store_location=vocab_store_location,
            debug=DEBUG,
            embedding_type="random",
            embedding_dimension=EMBEDDING_DIM,
            add_start_end_token=True,
        )

    options = {
        "MAX_NUM_WORDS": MAX_NUM_WORDS,
        "MAX_LENGTH": MAX_LENGTH,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }

    return train_dataset, test_dataset, options


class TestParscitDataset:
    def test_num_classes(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        num_classes = train_dataset.get_num_classes()
        assert num_classes == 16

    def test_lines_labels_not_empty(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        train_lines, train_labels = train_dataset.get_lines_labels()
        assert all([bool(line.strip()) for line in train_lines])
        assert all([bool(label.strip()) for label in train_labels])

        test_lines, test_labels = test_dataset.get_lines_labels()
        assert all([bool(line.strip()) for line in test_lines])
        assert all([bool(label.strip()) for label in test_labels])

    def test_lines_labels_are_equal_length(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        train_lines, train_labels = train_dataset.get_lines_labels()
        len_lines_labels = zip(
            (len(line.split()) for line in train_lines),
            (len(label.split()) for label in train_labels),
        )
        assert all([len_line == len_label for len_line, len_label in len_lines_labels])

        test_lines, test_labels = test_dataset.get_lines_labels()
        len_lines_labels = zip(
            (len(line.split()) for line in test_lines),
            (len(label.split()) for label in test_labels),
        )
        assert all([len_line == len_label for len_line, len_label in len_lines_labels])

    def test_get_stats_works(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        try:
            train_dataset.get_stats()
        except:
            pytest.fail(
                f"Get stats for Parscit {train_dataset.dataset_type} does not work"
            )

        try:
            test_dataset.get_stats()
        except:
            pytest.fail(
                f"Get stats for Parscit {train_dataset.dataset_type} does not work"
            )

    def test_tokens_max_length(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        num_lines = len(lines)
        for idx in range(num_lines):
            assert len(train_dataset[idx]["tokens"]) == options["MAX_LENGTH"]

    def test_instance_dict_with_loader(self, setup_parscit_train_dataset):
        train_dataset, test_dataset, dataset_options = setup_parscit_train_dataset
        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
        instances_dict = next(iter(loader))
        assert len(instances_dict["tokens"]) == 2

        test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
        instances_dict = next(iter(test_loader))
        assert len(instances_dict["tokens"]) == 2

    def test_instance_dict_have_correct_padded_lengths(
        self, setup_parscit_train_dataset
    ):
        train_dataset, test_dataset, options = setup_parscit_train_dataset
        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
        instances_dict = next(iter(loader))

        assert instances_dict["tokens"].size() == (2, options["MAX_LENGTH"])
        assert instances_dict["label"].size() == (2, options["MAX_LENGTH"])

    def test_labels_maxlen_2(self, setup_parscit_train_dataset_maxlen_2):
        train_dataset, test_dataset, options = setup_parscit_train_dataset_maxlen_2
        instances_dict = train_dataset[0]
        label = instances_dict["label"].tolist()
        tokens = instances_dict["tokens"].tolist()
        label = [train_dataset.idx2classname[lbl] for lbl in label]
        tokens = [train_dataset.word_vocab.idx2token[token_idx] for token_idx in tokens]
        assert label == ["starting", "ending"]
        assert tokens == ["<SOS>", "<EOS>"]
