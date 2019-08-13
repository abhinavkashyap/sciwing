import parsect.constants as constants
import pytest
from parsect.datasets.classification.parsect_dataset import ParsectDataset
import torch
from torch.utils.data import DataLoader


FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


@pytest.fixture(scope="session")
def setup_parsect_train_dataset(tmpdir_factory):
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


class TestParsectDataset:
    def test_label_mapping_len(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        label_mapping = train_dataset.get_classname2idx()
        assert len(label_mapping) == 23

    def test_no_line_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels(filename=SECT_LABEL_FILE)
        assert all([bool(line.strip()) for line in lines])

    def test_no_label_empty(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels(filename=SECT_LABEL_FILE)
        assert all([bool(label.strip()) for label in labels])

    def test_tokens_max_length(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        lines, labels = train_dataset.get_lines_labels(filename=SECT_LABEL_FILE)
        num_lines = len(lines)
        for idx in range(num_lines):
            assert len(train_dataset[idx]["tokens"]) == dataset_options["MAX_LENGTH"]

    def test_get_class_names_from_indices(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        instance_dict = next(iter(train_dataset))
        labels = instance_dict["label"]
        labels_list = labels.tolist()
        true_classnames = train_dataset.get_class_names_from_indices(labels_list)
        assert len(true_classnames) == len(labels_list)

    def test_get_disp_sentence_from_indices(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
        instances_dict = next(iter(loader))
        tokens = instances_dict["tokens"]
        tokens_list = tokens.tolist()
        train_sentence = train_dataset.word_vocab.get_disp_sentence_from_indices(
            tokens_list[0]
        )
        assert all([True for sentence in train_sentence if type(sentence) == str])

    def test_preloaded_embedding_has_values(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        preloaded_emb = train_dataset.get_preloaded_word_embedding()
        assert type(preloaded_emb) == torch.Tensor

    def test_dataset_returns_instances_when_required(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        first_instance = train_dataset[0]["instance"].split()
        assert all([type(word) == str for word in first_instance])

    def test_loader_returns_list_of_instances(self, setup_parsect_train_dataset):
        train_dataset, dataset_options = setup_parsect_train_dataset
        loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=False)
        instances_dict = next(iter(loader))

        assert len(instances_dict["instance"]) == 3
        assert type(instances_dict["instance"][0]) == str

    def test_stratified_split(self, setup_parsect_train_dataset):
        dataset, options = setup_parsect_train_dataset
        lines = ["a"] * 100
        labels = ["title"] * 100

        (
            (train_lines, train_labels),
            (validation_lines, validation_labels),
            (test_lines, test_labels),
        ) = dataset.get_train_valid_test_stratified_split(
            lines, labels, dataset.classname2idx
        )

        assert len(train_lines) == 80
        assert len(train_labels) == 80
        assert len(validation_lines) == 10
        assert len(validation_labels) == 10
        assert len(test_lines) == 10
        assert len(test_labels) == 10

    def test_get_lines_labels_stratified(self, setup_parsect_train_dataset):
        dataset, options = setup_parsect_train_dataset
        dataset.debug_dataset_proportion = 1

        lines, labels = dataset.get_lines_labels(filename=SECT_LABEL_FILE)
        assert all([bool(line.strip()) for line in lines])
        assert all([bool(label.strip()) for label in labels])

    def test_print_stats_works(self, setup_parsect_train_dataset):
        dataset, options = setup_parsect_train_dataset
        try:
            dataset.print_stats()
        except:
            pytest.fail("Test print stats works")
