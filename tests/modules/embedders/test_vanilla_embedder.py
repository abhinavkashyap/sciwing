import pytest
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
import torch.nn as nn
import torch


@pytest.fixture
def embedder():
    batch_size = 32
    time_steps = 10
    vocab_size = 3000
    embedding_dim = 300

    embedding = nn.Embedding.from_pretrained(
        embeddings=torch.rand(vocab_size, embedding_dim), freeze=False
    )
    tokens = torch.LongTensor(
        torch.randint(0, vocab_size, size=(batch_size, time_steps))
    )
    options = {
        "tokens": tokens,
        "embedding_size": embedding_dim,
        "batch_size": batch_size,
        "time_steps": time_steps,
    }
    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=embedding_dim)
    return embedder, options


class TestVanillaEmbedder:
    def test_dimension(self, embedder):
        embedder, options = embedder
        iter_dict = {"tokens": options["tokens"]}
        embedding_size = options["embedding_size"]
        batch_size = options["batch_size"]
        time_steps = options["time_steps"]

        embedding = embedder(iter_dict=iter_dict)
        assert embedding.size() == (batch_size, time_steps, embedding_size)
