import pytest
import sciwing.constants as constants
import pathlib
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.utils.common import chunks
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
import torch
from sciwing.utils.class_nursery import ClassNursery

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture(scope="session")
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


@pytest.fixture(
    scope="session",
    params=["glove_6B_50", "glove_6B_100", "glove_6B_200", "glove_6B_300", "parscit"],
)
def setup_embedder(setup_parscit_dataset_manager, request):
    data_manager = setup_parscit_dataset_manager
    embedding_type = request.param
    embedder = TrainableWordEmbedder(
        datasets_manager=data_manager, embedding_type=embedding_type
    )
    return embedder, data_manager


class TestTrainableWordEmbedder:
    def test_returns_float_tensors(self, setup_embedder):
        embedder, data_manager = setup_embedder
        train_dataset = data_manager.train_dataset
        lines, labels = train_dataset.get_lines_labels()
        for lines_batch in chunks(lines, 10):
            embedding = embedder(lines_batch)
            assert isinstance(embedding, torch.FloatTensor)

    def test_module_has_trainable_params(self, setup_embedder):
        embedder, data_manager = setup_embedder
        for param in embedder.parameters():
            assert param.requires_grad

    def test_embedder_in_class_nursery(self):
        assert ClassNursery.class_nursery["TrainableWordEmbedder"] is not None

    def test_embedding_dimensions(self, setup_embedder):
        embedder, data_manager = setup_embedder
        train_dataset = data_manager.train_dataset
        lines, labels = train_dataset.get_lines_labels()
        for lines_batch in chunks(lines, 10):
            embedding = embedder(lines_batch)
            assert embedding.dim() == 3
