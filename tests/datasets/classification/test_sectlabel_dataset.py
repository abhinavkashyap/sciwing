import sciwing.constants as constants
import pytest
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pathlib

FILES = constants.PATHS
DATA_DIR = FILES["DATA_DIR"]


@pytest.fixture(scope="module")
def setup_sectlabel_dataset_manager():
    data_dir = pathlib.Path(DATA_DIR)
    sect_label_train_file = data_dir.joinpath("sectLabel.train")
    sect_label_dev_file = data_dir.joinpath("sectLabel.dev")
    sect_label_test_file = data_dir.joinpath("sectLabel.test")

    dataset_manager = TextClassificationDatasetManager(
        train_filename=sect_label_train_file,
        dev_filename=sect_label_dev_file,
        test_filename=sect_label_test_file,
    )

    return dataset_manager


class TestSectLabelDataset:
    def test_label_mapping_len(self, setup_sectlabel_dataset_manager):
        dataset_manager = setup_sectlabel_dataset_manager
        train_dataset = dataset_manager.train_dataset
        lines, labels = train_dataset.get_lines_labels()
        labels = [label.text for label in labels]
        labels = list(set(labels))
        assert len(labels) == 23

    def test_no_line_empty(self, setup_sectlabel_dataset_manager):
        dataset_manager = setup_sectlabel_dataset_manager
        for dataset in [
            dataset_manager.train_dataset,
            dataset_manager.dev_dataset,
            dataset_manager.test_dataset,
        ]:
            lines, labels = dataset.get_lines_labels()
            assert all([bool(line.text.strip()) for line in lines])

    def test_no_label_empty(self, setup_sectlabel_dataset_manager):
        dataset_manager = setup_sectlabel_dataset_manager
        for dataset in [
            dataset_manager.train_dataset,
            dataset_manager.dev_dataset,
            dataset_manager.test_dataset,
        ]:
            lines, labels = dataset.get_lines_labels()
            assert all([bool(label.text.strip()) for label in labels])
