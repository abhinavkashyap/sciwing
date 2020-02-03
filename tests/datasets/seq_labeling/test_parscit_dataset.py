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
            train_labels.extend(label.labels)
        num_classes = len(set(train_labels))
        assert num_classes == 14
