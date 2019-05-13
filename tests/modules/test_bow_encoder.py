import numpy as np
import pytest
import torch
import torch.nn as nn
from parsect.modules.bow_encoder import BOW_Encoder


@pytest.fixture
def setup_zero_embeddings_test_bs_1_sum():
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    embedding = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM,
        embedding=embedding,
    )
    tokens = np.random.randint(0, VOCAB_SIZE-1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    options = {'EMB_DIM': EMB_DIM, 'VOCAB_SIZE': VOCAB_SIZE, 'BATCH_SIZE': BATCH_SIZE}
    return encoder, tokens, options


@pytest.fixture
def setup_zero_embeddings_test_bs_1_average():
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    embedding = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM,
        embedding=embedding,
        aggregation_type='average'
    )
    tokens = np.random.randint(0, VOCAB_SIZE-1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    options = {'EMB_DIM': EMB_DIM, 'VOCAB_SIZE': VOCAB_SIZE, 'BATCH_SIZE': BATCH_SIZE}
    return encoder, tokens, options


@pytest.fixture
def setup_zero_embeddings_test_bs_10_average():
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 10
    embedding = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM,
        embedding=embedding,
        aggregation_type='average'
    )
    tokens = np.random.randint(0, VOCAB_SIZE-1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    options = {'EMB_DIM': EMB_DIM, 'VOCAB_SIZE': VOCAB_SIZE, 'BATCH_SIZE': BATCH_SIZE}
    return encoder, tokens, options


@pytest.fixture
def setup_one_embeddings_test_bs_1_sum():
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    embedding = nn.Embedding.from_pretrained(torch.ones([VOCAB_SIZE, EMB_DIM]))
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM,
        embedding=embedding,
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    options = {'EMB_DIM': EMB_DIM, 'VOCAB_SIZE': VOCAB_SIZE, 'BATCH_SIZE': BATCH_SIZE}
    return encoder, tokens, options


@pytest.fixture
def setup_one_embeddings_test_bs_1_average():
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 1
    embedding = nn.Embedding.from_pretrained(torch.ones([VOCAB_SIZE, EMB_DIM]))
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM,
        embedding=embedding,
        aggregation_type='average'
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    options = {'EMB_DIM': EMB_DIM, 'VOCAB_SIZE': VOCAB_SIZE, 'BATCH_SIZE': BATCH_SIZE}
    return encoder, tokens, options


class TestBOWEncoder:
    def test_bow_encoder_zeros_sum(self, setup_zero_embeddings_test_bs_1_sum):
        encoder, tokens, options = setup_zero_embeddings_test_bs_1_sum
        BATCH_SIZE = options['BATCH_SIZE']
        EMB_DIM = options['EMB_DIM']

        embeddings = encoder(tokens)
        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, torch.zeros([BATCH_SIZE, EMB_DIM]))).item()

    def test_bow_encoder_zeros_sum_bs_10(self, setup_zero_embeddings_test_bs_10_average):
        encoder, tokens, options = setup_zero_embeddings_test_bs_10_average
        BATCH_SIZE = options['BATCH_SIZE']
        EMB_DIM = options['EMB_DIM']

        embeddings = encoder(tokens)
        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, torch.zeros([BATCH_SIZE, EMB_DIM]))).item()

    def test_bow_encoder_zeros_average(self, setup_zero_embeddings_test_bs_1_average):
        encoder, tokens, options = setup_zero_embeddings_test_bs_1_average
        BATCH_SIZE = options['BATCH_SIZE']
        EMB_DIM = options['EMB_DIM']

        embeddings = encoder(tokens)
        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, torch.zeros([BATCH_SIZE, EMB_DIM]))).item()

    def test_bow_encoder_ones_sum(self, setup_one_embeddings_test_bs_1_sum):
        encoder, tokens, options = setup_one_embeddings_test_bs_1_sum
        BATCH_SIZE = options['BATCH_SIZE']
        EMB_DIM = options['EMB_DIM']

        embeddings = encoder(tokens)
        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, torch.ones([BATCH_SIZE, EMB_DIM]) * EMB_DIM)).item()

    def test_bow_encoder_ones_average(self, setup_one_embeddings_test_bs_1_average):
        encoder, tokens, options = setup_one_embeddings_test_bs_1_average
        BATCH_SIZE = options['BATCH_SIZE']
        EMB_DIM = options['EMB_DIM']

        embeddings = encoder(tokens)
        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, torch.ones([BATCH_SIZE, EMB_DIM]))).item()
