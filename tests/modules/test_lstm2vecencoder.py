import pytest
import torch
import torch.nn as nn
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
import numpy as np
import itertools

directions = [True, False]  # True for bi-directions
combination_strategy = ["concat", "sum"]
have_additional_embedding_ = [True, False]

directions_combine_strategy = itertools.product(
    directions, combination_strategy, have_additional_embedding_
)
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
    have_additional_embedding = request.param[2]
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, time_steps))
    tokens = torch.LongTensor(tokens)
    additional_embedding = None

    if have_additional_embedding:
        additional_embedding = torch.zeros(batch_size, time_steps, emb_dim)
        emb_dim += emb_dim

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedding=embedding,
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
            "have_additional_embedding": have_additional_embedding,
            "additional_embedding": additional_embedding,
        },
    )


class TestLstm2VecEncoder:
    def test_raises_error_on_wrong_combine_strategy(self):
        with pytest.raises(AssertionError):
            encoder = LSTM2VecEncoder(
                emb_dim=300, embedding=nn.Embedding(10, 1024), combine_strategy="add"
            )

    def test_zero_embedding_dimension(self, setup_lstm2vecencoder):
        encoder, options = setup_lstm2vecencoder
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x, additional_embedding=options["additional_embedding"])
        assert encoding.size() == (batch_size, hidden_dim)

    def test_lstm2vec_encoder(self, setup_lstm2vecencoder):
        encoder, options = setup_lstm2vecencoder
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x, additional_embedding=options["additional_embedding"])
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, hidden_dim]))
        ).item()
