import pytest
import torch
import torch.nn as nn
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
import numpy as np


@pytest.fixture
def setup_unidirection_concat_zero_embedding():
    emb_dim = 300
    vocab_size = 100
    batch_size = 32
    embedding = nn.Embedding.from_pretrained(torch.zeros([vocab_size, emb_dim]))
    hidden_dimension = 1024
    combine_strategy = "concat"
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, emb_dim))
    tokens = torch.LongTensor(tokens)

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=False,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    return (
        encoder,
        {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "tokens": tokens,
            "batch_size": batch_size,
        },
    )


@pytest.fixture
def setup_unidirection_sum_zero_embedding():
    emb_dim = 300
    vocab_size = 100
    batch_size = 32
    embedding = nn.Embedding.from_pretrained(torch.zeros([vocab_size, emb_dim]))
    hidden_dimension = 1024
    combine_strategy = "sum"
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, emb_dim))
    tokens = torch.LongTensor(tokens)

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=False,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    return (
        encoder,
        {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "tokens": tokens,
            "batch_size": batch_size,
        },
    )


@pytest.fixture
def setup_bidirection_concat_zero_embedding():
    emb_dim = 300
    vocab_size = 100
    batch_size = 32
    embedding = nn.Embedding.from_pretrained(torch.zeros([vocab_size, emb_dim]))
    hidden_dimension = 1024
    combine_strategy = "concat"
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, emb_dim))
    tokens = torch.LongTensor(tokens)

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=True,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    return (
        encoder,
        {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "tokens": tokens,
            "batch_size": batch_size,
        },
    )


@pytest.fixture
def setup_bidirection_sum_zero_embedding():
    emb_dim = 300
    vocab_size = 100
    batch_size = 32
    embedding = nn.Embedding.from_pretrained(torch.zeros([vocab_size, emb_dim]))
    hidden_dimension = 1024
    combine_strategy = "sum"
    tokens = np.random.randint(0, vocab_size - 1, size=(batch_size, emb_dim))
    tokens = torch.LongTensor(tokens)

    encoder = LSTM2VecEncoder(
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=True,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    return (
        encoder,
        {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "tokens": tokens,
            "batch_size": batch_size,
        },
    )


class TestLstm2VecEncoder:
    def test_raises_error_on_wrong_combine_strategy(self):
        with pytest.raises(AssertionError):
            encoder = LSTM2VecEncoder(
                emb_dim=300, embedding=nn.Embedding(10, 1024), combine_strategy="add"
            )

    def test_zero_embedding_unidir_concat_dimension(
        self, setup_unidirection_concat_zero_embedding
    ):
        encoder, options = setup_unidirection_concat_zero_embedding
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x)
        assert encoding.size() == (batch_size, hidden_dim)

    def test_zero_embedding_unidir_concat(
        self, setup_unidirection_concat_zero_embedding
    ):
        encoder, options = setup_unidirection_concat_zero_embedding
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x)
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, hidden_dim]))
        ).item()

    def test_zero_embedding_unidir_sum_dimensions(
        self, setup_unidirection_sum_zero_embedding
    ):
        encoder, options = setup_unidirection_sum_zero_embedding
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x)
        assert encoding.size() == (batch_size, hidden_dim)

    def test_zero_embedding_unidir_sum(self, setup_unidirection_concat_zero_embedding):
        encoder, options = setup_unidirection_concat_zero_embedding
        x = options["tokens"]
        hidden_dim = options["hidden_dim"]
        batch_size = options["batch_size"]
        encoding = encoder(x)
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, hidden_dim]))
        ).item()

    def test_zero_embedding_bidir_concat_dimension(
        self, setup_bidirection_concat_zero_embedding
    ):
        encoder, options = setup_bidirection_concat_zero_embedding
        batch_size = options["batch_size"]
        hidden_dim = options["hidden_dim"]
        x = options["tokens"]

        encoding = encoder(x)
        assert encoding.size() == (batch_size, 2 * hidden_dim)

    def test_zero_embedding_bidir_concat(self, setup_bidirection_concat_zero_embedding):
        encoder, options = setup_bidirection_concat_zero_embedding
        batch_size = options["batch_size"]
        hidden_dim = options["hidden_dim"]
        x = options["tokens"]

        encoding = encoder(x)
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, 2 * hidden_dim]))
        ).item()

    def test_zero_embedding_bidir_sum_dimension(
        self, setup_bidirection_sum_zero_embedding
    ):
        encoder, options = setup_bidirection_sum_zero_embedding
        batch_size = options["batch_size"]
        hidden_dim = options["hidden_dim"]
        x = options["tokens"]

        encoding = encoder(x)
        assert encoding.size() == (batch_size, 1 * hidden_dim)

    def test_zero_embedding_bidir_sum(self, setup_bidirection_sum_zero_embedding):
        encoder, options = setup_bidirection_sum_zero_embedding
        batch_size = options["batch_size"]
        hidden_dim = options["hidden_dim"]
        x = options["tokens"]

        encoding = encoder(x)
        assert torch.all(
            torch.eq(encoding, torch.zeros([batch_size, 1 * hidden_dim]))
        ).item()
