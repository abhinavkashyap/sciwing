import pytest
from sciwing.vocab.embedding_loader import EmbeddingLoader
import numpy as np
import os
from sciwing.utils.common import get_system_mem_in_gb
import pathlib
import sciwing.constants as constants
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
import torch

DATA_DIR = constants.PATHS["DATA_DIR"]


@pytest.fixture(
    params=[
        "glove_6B_50",
        "glove_6B_100",
        "glove_6B_200",
        "glove_6B_300",
        "parscit",
        "lample_conll",
    ],
    scope="session",
)
def setup_word_emb_loader(request):
    embedding_type = request.param
    embedding_loader = EmbeddingLoader(embedding_type=embedding_type)
    return embedding_loader


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


memory_available = int(get_system_mem_in_gb())


@pytest.mark.skipif(
    memory_available < 16, reason="Memory is too low to run the word emb loader tests"
)
class TestWordEmbLoader:
    def test_invalid_embedding_type(self):
        with pytest.raises(AssertionError):
            loader = EmbeddingLoader(embedding_type="nonexistent")

    def test_preloaded_file_exists(self, setup_word_emb_loader):
        emb_loader = setup_word_emb_loader
        preloaded_filename = emb_loader.get_preloaded_filename()

        assert os.path.isfile(preloaded_filename)

    def test_embeddings_are_np_arrays(self, setup_word_emb_loader):

        emb_loader = setup_word_emb_loader
        if emb_loader.embedding_type != "parscit":
            for word, embedding in emb_loader._embeddings.items():
                assert isinstance(embedding, np.ndarray)

    def test_get_embedding_for_vocab_returns_tensor(
        self, setup_word_emb_loader, setup_parscit_dataset_manager
    ):
        emb_loader = setup_word_emb_loader
        data_manager = setup_parscit_dataset_manager
        vocab = data_manager.namespace_to_vocab["tokens"]
        embedding = emb_loader.get_embeddings_for_vocab(vocab=vocab)
        assert isinstance(embedding, torch.FloatTensor)

    def test_get_embedding_for_vocab_length(
        self, setup_word_emb_loader, setup_parscit_dataset_manager
    ):
        emb_loader = setup_word_emb_loader
        data_manager = setup_parscit_dataset_manager
        vocab = data_manager.namespace_to_vocab["tokens"]
        vocab_len = vocab.get_vocab_len()
        embedding = emb_loader.get_embeddings_for_vocab(vocab=vocab)
        assert embedding.size(0) == vocab_len
