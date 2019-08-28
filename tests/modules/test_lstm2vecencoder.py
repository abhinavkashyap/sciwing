import pytest
import torch
import torch.nn as nn
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
import numpy as np
import itertools

directions = [True, False]  # True for bi-directions
combination_strategy = ["concat", "sum"]

directions_combine_strategy = itertools.product(directions, combination_strategy)
directions_combine_strategy = list(directions_combine_strategy)


@pytest.fixture(params=directions_combine_strategy)
def setup_lstm2vecencoder(request):
    emb_dim = 300
    time_steps = 10
    vocab_size = 100
    batch_size = 32
    embedding = nn.Embedding.from_pretrained(torch.zeros([vocab_size, emb_dim]))
    hidden_dimension = 1024
    combine_strategy = request.param[1]
    bidirectional = request.param[0]
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, time_steps))
    tokens = torch.LongTensor(tokens)

    iter_dict = {"tokens": tokens}
    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=emb_dim)

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=bidirectional,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    return (
        encoder,
        {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "hidden_dim": 2 * hidden_dimension
            if bidirectional and combine_strategy == "concat"
            else hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "tokens": tokens,
            "batch_size": batch_size,
            "iter_dict": iter_dict,
        },
    )


class TestLstm2VecEncoder:
    def test_raises_error_on_wrong_combine_strategy(self, setup_lstm2vecencoder):
        with pytest.raises(AssertionError):

            encoder = LSTM2VecEncoder(
                emb_dim=300,
                embedder=VanillaEmbedder(nn.Embedding(10, 1024), embedding_dim=1024),
                combine_strategy="add",
            )

    def test_zero_embedding_dimension(self, setup_lstm2vecencoder):
        encoder, options = setup_lstm2vecencoder
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(iter_dict=options["iter_dict"])
        assert encoding.size() == (batch_size, hidden_dim)

    def test_lstm2vec_encoder(self, setup_lstm2vecencoder):
        encoder, options = setup_lstm2vecencoder
        iter_dict = options["iter_dict"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(iter_dict=iter_dict)
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, hidden_dim]))
        ).item()
