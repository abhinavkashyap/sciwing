import pytest
import sciwing.constants as constants
from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
from sciwing.utils.class_nursery import ClassNursery
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR = pathlib.Path(DATA_DIR)


@pytest.fixture
def setup_science_ie_dataset():
    train_filename = DATA_DIR.joinpath("train_science_ie_conll.txt")
    dev_filename = DATA_DIR.joinpath("dev_science_ie_conll.txt")
    data_manager = CoNLLDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=dev_filename,
        column_names=["TASK", "PROCESS", "MATERIAL"],
    )
    return data_manager


class TestScienceIE:
    def test_label_namespaces(self, setup_science_ie_dataset):
        data_manager = setup_science_ie_dataset
        label_namespaces = data_manager.label_namespaces
        assert "TASK" in label_namespaces
        assert "PROCESS" in label_namespaces
        assert "MATERIAL" in label_namespaces

    def test_num_classes(self, setup_science_ie_dataset):
        data_manager = setup_science_ie_dataset
        label_namespaces = data_manager.label_namespaces
        for namespace in label_namespaces:
            assert data_manager.num_labels[namespace] == 9

    def test_lines_labels_not_empty(self, setup_science_ie_dataset):
        data_manager = setup_science_ie_dataset
        lines, labels = data_manager.train_dataset.get_lines_labels()

        for line, label in zip(lines, labels):
            line_text = line.text
            task_label_tokens = label.tokens["TASK"]
            process_label_tokens = label.tokens["PROCESS"]
            material_label_tokens = label.tokens["MATERIAL"]

            task_label_tokens = [tok.text for tok in task_label_tokens]
            process_label_tokens = [tok.text for tok in process_label_tokens]
            material_label_tokens = [tok.text for tok in material_label_tokens]

            assert bool(line_text.strip())
            assert all([bool(tok) for tok in task_label_tokens])
            assert all([bool(tok) for tok in process_label_tokens])
            assert all([bool(tok) for tok in material_label_tokens])

    def test_lines_labels_are_equal_length(self, setup_science_ie_dataset):
        data_manager = setup_science_ie_dataset
        lines, labels = data_manager.train_dataset.get_lines_labels()

        for line, label in zip(lines, labels):
            line_tokens = line.tokens["tokens"]
            line_tokens = [tok.text for tok in line_tokens]
            for namespace in ["TASK", "PROCESS", "MATERIAL"]:
                label_tokens = label.tokens[namespace]
                label_tokens = [tok.text for tok in label_tokens]
                assert len(line_tokens) == len(label_tokens)

    def test_science_ie_in_nursery(self):
        assert ClassNursery.class_nursery.get("CoNLLDatasetManager") is not None
