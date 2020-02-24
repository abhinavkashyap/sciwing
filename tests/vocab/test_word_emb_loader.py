import pytest
from sciwing.vocab.embedding_loader import EmbeddingLoader
import numpy as np
import os
from sciwing.utils.common import get_system_mem_in_gb


@pytest.fixture(
    params=["glove_6B_50", "glove_6B_100", "glove_6B_200", "glove_6B_300", "parscit"],
    scope="session",
)
def setup_word_emb_loader(request):
    embedding_type = request.param
    embedding_loader = EmbeddingLoader(embedding_type=embedding_type)
    return embedding_loader


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
