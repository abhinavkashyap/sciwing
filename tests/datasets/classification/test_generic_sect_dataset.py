from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import pytest
import sciwing.constants as constants
import pathlib

FILES = constants.FILES
PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]


@pytest.fixture(scope="module")
def setup_generic_sect_dataset_manager():
    data_dir = pathlib.Path(DATA_DIR)
    sect_label_train_file = data_dir.joinpath("genericSect.train")
    sect_label_dev_file = data_dir.joinpath("genericSect.dev")
    sect_label_test_file = data_dir.joinpath("genericSect.test")

    dataset_manager = TextClassificationDatasetManager(
        train_filename=sect_label_train_file,
        dev_filename=sect_label_dev_file,
        test_filename=sect_label_test_file,
    )

    return dataset_manager


class TestGenericSectDataset:
    def test_num_labels(self, setup_generic_sect_dataset_manager):
        dataset_manager = setup_generic_sect_dataset_manager
        train_dataset = dataset_manager.train_dataset
        lines, labels = train_dataset.get_lines_labels()
        labels = [label.text for label in labels]
        num_labels = len(set(labels))
        assert num_labels == 12

    def test_no_line_empty(self, setup_generic_sect_dataset_manager):
        dataset_manager = setup_generic_sect_dataset_manager
        for dataset in [
            dataset_manager.train_dataset,
            dataset_manager.dev_dataset,
            dataset_manager.test_dataset,
        ]:
            lines, labels = dataset.get_lines_labels()
            assert all([bool(line.text.strip()) for line in lines])

    def test_no_train_label_empty(self, setup_generic_sect_dataset_manager):
        dataset_manager = setup_generic_sect_dataset_manager
        for dataset in [
            dataset_manager.train_dataset,
            dataset_manager.dev_dataset,
            dataset_manager.test_dataset,
        ]:
            lines, labels = dataset.get_lines_labels()
            assert all([bool(label.text.strip()) for label in labels])
