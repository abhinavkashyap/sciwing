import pytest
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
from parsect.modules.elmo_embedder import ElmoEmbedder
import torch
import torch.nn as nn
import numpy as np


@pytest.fixture
def setup_test_elmo_lstm_encoder_concat():
    elmo_embedder = ElmoEmbedder()
    elmo_emb_dim = 1024
    emb_dim = 100
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    NUM_TOKENS = 5
    hidden_dim = 1024
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "concat"
    RNN_BIAS = True
    instances = ["i like to test this".split()]
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor([[1]])
    embedding = torch.nn.Embedding(VOCAB_SIZE, emb_dim)

    elmo_lstm_encoder = ElmoLSTMEncoder(
        elmo_emb_dim=elmo_emb_dim,
        elmo_embedder=elmo_embedder,
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dim,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=RNN_BIAS,
    )

    return (
        tokens,
        labels,
        instances,
        elmo_lstm_encoder,
        {
            "ELMO_EMB_DIM": 1024,
            "EMB_DIM": emb_dim,
            "VOCAB_SIZE": VOCAB_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "RNN_BIAS": RNN_BIAS,
            "HIDDEN_DIM": hidden_dim,
        },
    )


@pytest.fixture
def setup_test_elmo_lstm_encoder_sum():
    elmo_embedder = ElmoEmbedder()
    elmo_emb_dim = 1024
    emb_dim = 100
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    NUM_TOKENS = 5
    hidden_dim = 1024
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "sum"
    RNN_BIAS = True
    instances = ["i like to test this".split()]
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor([[1]])
    embedding = torch.nn.Embedding(VOCAB_SIZE, emb_dim)

    elmo_lstm_encoder = ElmoLSTMEncoder(
        elmo_emb_dim=elmo_emb_dim,
        elmo_embedder=elmo_embedder,
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=hidden_dim,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=RNN_BIAS,
    )

    return (
        tokens,
        labels,
        instances,
        elmo_lstm_encoder,
        {
            "ELMO_EMB_DIM": 1024,
            "EMB_DIM": emb_dim,
            "VOCAB_SIZE": VOCAB_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "RNN_BIAS": RNN_BIAS,
            "HIDDEN_DIM": hidden_dim,
        },
    )


class TestElmoLSTMEncoder:
    def test_encoding_dimension_concat(self, setup_test_elmo_lstm_encoder_concat):
        tokens, labels, instances, elmo_lstm_encoder, options = (
            setup_test_elmo_lstm_encoder_concat
        )
        encoding = elmo_lstm_encoder(x=tokens, instances=instances)
        batch_size = options["BATCH_SIZE"]
        hidden_dim = options["HIDDEN_DIM"]

        # bidirectional concat embedding
        assert encoding.size() == (batch_size, 2 * hidden_dim)

    def test_encoding_dimension_sum(self, setup_test_elmo_lstm_encoder_sum):
        tokens, labels, instances, elmo_lstm_encoder, options = (
            setup_test_elmo_lstm_encoder_sum
        )
        encoding = elmo_lstm_encoder(x=tokens, instances=instances)
        batch_size = options["BATCH_SIZE"]
        hidden_dim = options["HIDDEN_DIM"]

        # bidirectional concat embedding
        assert encoding.size() == (batch_size, hidden_dim)
