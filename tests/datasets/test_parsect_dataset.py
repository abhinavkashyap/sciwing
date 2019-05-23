import pytest
import parsect.constants as constants
import pytest
from parsect.datasets.parsect_dataset import ParsectDataset
import torch
from torch.utils.data import DataLoader

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


@pytest.fixture
def setup_parsect_train_dataset(tmpdir):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 10
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
    )

    return (
        train_dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "vocab_store_location": vocab_store_location,
        },
    )


@pytest.fixture
def setup_parsect_train_dataset_returns_instances(tmpdir):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 10
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        return_instances=True,
    )

    return (
        train_dataset,
        {
            "MAX_NUM_WORDS": MAX_NUM_WORDS,
            "MAX_LENGTH": MAX_LENGTH,
            "vocab_store_location": vocab_store_location,
        },
    )


class TestParsectDataset:
    def test_label_mapping_len(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        label_mapping = train_dataset.get_label_mapping()
        assert len(label_mapping) == 23

    def test_no_line_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        assert all([line != "" for line in lines])

    def test_no_label_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        assert all([label != "" for label in labels])

    def test_tokens_max_length(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels()
        num_lines = len(lines)
        for idx in range(num_lines):
            assert len(train_dataset[idx][0]) == dataset_options["MAX_LENGTH"]

    def test_get_class_names_from_indices(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        tokens, labels, len_tokens = next(iter(train_dataset))
        labels_list = labels.tolist()
        true_classnames = train_dataset.get_class_names_from_indices(labels_list)
        assert len(true_classnames) == len(labels_list)

    def test_get_disp_sentence_from_indices(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
        tokens, labels, len_tokens = next(iter(loader))
        tokens_list = tokens.tolist()
        train_sentence = train_dataset.get_disp_sentence_from_indices(tokens_list[0])
        assert all([True for sentence in train_sentence if type(sentence) == str])

    def test_preloaded_embedding_has_values(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        preloaded_emb = train_dataset.get_preloaded_embedding()
        assert type(preloaded_emb) == torch.Tensor

    def test_dataset_returns_instances_when_required(
        self, setup_parsect_train_dataset_returns_instances
    ):
        train_dataset, dataset_options = setup_parsect_train_dataset_returns_instances
        first_instance = train_dataset[0][0]
        assert all([type(word) == str for word in first_instance])

    def test_loader_returns_list_of_instances(
        self, setup_parsect_train_dataset_returns_instances
    ):
        def collate_fn(batch):
            instances = []
            labels = []
            len_tokens = []

            for ele in batch:
                instances.append(ele[0])
                labels.append(ele[1])
                len_tokens.append(ele[2])

            labels = torch.stack(labels, dim=0)
            len_tokens = torch.stack(len_tokens, dim=0)

            return instances, labels, len_tokens

        train_dataset, dataset_options = setup_parsect_train_dataset_returns_instances
        loader = DataLoader(
            dataset=train_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn
        )
        instances = next(iter(loader))

        assert len(instances) == 3
        assert type(instances[0]) == list
