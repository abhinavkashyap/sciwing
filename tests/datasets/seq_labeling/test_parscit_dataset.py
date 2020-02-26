import sciwing.constants as constants
import pytest
import pathlib
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)

FILES = constants.FILES
PATHS = constants.PATHS
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture
def setup_parscit_dataset_manager():
    data_dir = pathlib.Path(DATA_DIR)
    parscit_train_file = data_dir.joinpath("parscit.train")
    parscit_dev_file = data_dir.joinpath("parscit.dev")
    parscit_test_file = data_dir.joinpath("parscit.test")

    dataset_manager = SeqLabellingDatasetManager(
        train_filename=str(parscit_train_file),
        dev_filename=str(parscit_dev_file),
        test_filename=str(parscit_test_file),
    )
    return dataset_manager


class TestParscitDataset:
    def test_num_classes(self, setup_parscit_dataset_manager):
        dataset_manager = setup_parscit_dataset_manager
        train_dataset = dataset_manager.train_dataset
        lines, labels = train_dataset.get_lines_labels()
        train_labels = []
        for label in labels:
            label_tokens = label.tokens["seq_label"]
            label_tokens = [tok.text for tok in label_tokens]
            train_labels.extend(label_tokens)
        train_labels = set(train_labels)
        num_classes = len(train_labels)
        assert num_classes == 14

    def test_num_lines(self, setup_parscit_dataset_manager):
        dataset_manager = setup_parscit_dataset_manager
        train_dataset = dataset_manager.train_dataset
        dev_dataset = dataset_manager.dev_dataset
        test_dataset = dataset_manager.test_dataset
        assert len(train_dataset) == 1245
        assert len(dev_dataset) == 139
        assert len(test_dataset) == 139

    def test_lines_labels_equal_length(self, setup_parscit_dataset_manager):
        dataset_manager = setup_parscit_dataset_manager
        datasets = [
            dataset_manager.train_dataset,
            dataset_manager.dev_dataset,
            dataset_manager.test_dataset,
        ]
        for dataset in datasets:
            lines, labels = dataset.get_lines_labels()
            for line, label in zip(lines, labels):
                assert len(line.tokens["tokens"]) == len(label.tokens["seq_label"])
