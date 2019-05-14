import pytest
import parsect.constants as constants
import pytest
from parsect.datasets.parsect_dataset import ParsectDataset

FILES = constants.FILES
SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']


@pytest.fixture
def setup_parsect_train_dataset(tmpdir):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 10
    vocab_store_location = tmpdir.mkdir("tempdir").join('vocab.json')
    DEBUG = True

    train_dataset = ParsectDataset(secthead_label_file=SECT_LABEL_FILE,
                                   dataset_type='train',
                                   max_num_words=MAX_NUM_WORDS,
                                   max_length=MAX_LENGTH,
                                   vocab_store_location=vocab_store_location,
                                   debug=DEBUG)

    return train_dataset, {
        'MAX_NUM_WORDS': MAX_NUM_WORDS,
        'MAX_LENGTH': MAX_LENGTH,
        'vocab_store_location': vocab_store_location
    }


class TestParsectDataset:
    def test_label_mapping_len(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        label_mapping = train_dataset.get_label_mapping()
        assert len(label_mapping) == 23

    def test_no_line_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        assert all([line != '' for line in lines])

    def test_no_label_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        assert all([label != '' for label in labels])

    def test_tokens_max_length(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        num_lines = len(lines)
        for idx in range(num_lines):
            assert len(train_dataset[idx][0]) == dataset_options['MAX_LENGTH']


