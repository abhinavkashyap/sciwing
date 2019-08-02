import pytest
from parsect.modules.charlstm_encoder import CharLSTMEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
import torch.nn as nn
import torch


@pytest.fixture
def char_lstm_encoder():
    VOCAB_SIZE = 150
    EMBEDDING_DIM = 25
    BATCH_SIZE = 32
    NUM_TIME_STEPS = 10
    MAX_CHAR_SIZE = 5
    HIDDEN_DIM = 100
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    vanilla_embedder = VanillaEmbedder(embedding=embedding, embedding_dim=EMBEDDING_DIM)
    tokens = torch.randint(
        0, VOCAB_SIZE, size=(BATCH_SIZE, NUM_TIME_STEPS, MAX_CHAR_SIZE)
    )
    iter_dict = {"char_tokens": tokens}

    char_lstm_encoder = CharLSTMEncoder(
        char_embedder=vanilla_embedder,
        char_emb_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        bidirectional=True,
        combine_strategy="concat",
    )

    return (
        char_lstm_encoder,
        {
            "iter_dict": iter_dict,
            "VOCAB_SIZE": VOCAB_SIZE,
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_TIME_STEPS": NUM_TIME_STEPS,
            "MAX_CHAR_SIZE": MAX_CHAR_SIZE,
            "HIDDEN_DIM": HIDDEN_DIM,
            "EXPECTED_HIDDEN_DIM": HIDDEN_DIM * 2,
        },
    )


class TestCharLSTMEncoder:
    def test_char_lstm_encoder_dim(self, char_lstm_encoder):
        encoder, options = char_lstm_encoder
        iter_dict = options["iter_dict"]
        BATCH_SIZE = options["BATCH_SIZE"]
        NUM_TIME_STEPS = options["NUM_TIME_STEPS"]
        EXPECTED_HIDDEN_DIM = options["EXPECTED_HIDDEN_DIM"]
        embedding = encoder(iter_dict=iter_dict)
        assert embedding.size() == (BATCH_SIZE, NUM_TIME_STEPS, EXPECTED_HIDDEN_DIM)
