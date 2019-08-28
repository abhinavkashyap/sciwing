import pytest
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
import torch.nn as nn
import copy
import torch


@pytest.fixture()
def vanilla_embedder():
    embedding = nn.Embedding(1000, 100)
    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=100)
    return embedder, {"EMBEDDING_DIM": 100, "VOCAB_SIZE": 1000}


@pytest.fixture()
def bow_elmo_embedder():
    embedder = BowElmoEmbedder()
    instance = ["This is a string"]
    return embedder, {"instance": instance}


@pytest.fixture
def concat_vanilla_embedders(vanilla_embedder):
    first_embedder, options = vanilla_embedder
    second_embedder = copy.deepcopy(first_embedder)
    third_embedder = copy.deepcopy(second_embedder)
    EMBEDDING_DIM = options["EMBEDDING_DIM"]
    VOCAB_SIZE = options["VOCAB_SIZE"]
    BATCH_SIZE = 32
    MAX_TIME_STEPS = 10
    EXPECTED_EMBEDDING_DIM = EMBEDDING_DIM * 3
    concat_embedder = ConcatEmbedders([first_embedder, second_embedder, third_embedder])
    iter_dict = {
        "tokens": torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, MAX_TIME_STEPS))
    }
    return (
        concat_embedder,
        {
            "EXPECTED_EMBEDDING_DIM": EXPECTED_EMBEDDING_DIM,
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_TIME_STEPS": MAX_TIME_STEPS,
            "iter_dict": iter_dict,
        },
    )


@pytest.fixture
def concat_vanilla_bow_elmo_embedders(vanilla_embedder, bow_elmo_embedder):
    vanilla_word_embedder, vanilla_embedder_options = vanilla_embedder
    bow_elmo_word_embedder, bow_elmo_embedder_options = bow_elmo_embedder
    embedders = [vanilla_word_embedder, bow_elmo_word_embedder]
    VANILLA_EMBEDDING_DIM = vanilla_embedder_options["EMBEDDING_DIM"]
    VOCAB_SIZE = vanilla_embedder_options["VOCAB_SIZE"]
    BATCH_SIZE = 1
    MAX_TIME_STEPS = 4
    instance = bow_elmo_embedder_options["instance"]
    EXPECTED_EMBEDDING_DIM = VANILLA_EMBEDDING_DIM + 1024
    concat_embedder = ConcatEmbedders(embedders)
    iter_dict = {
        "tokens": torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, MAX_TIME_STEPS)),
        "instance": instance,
    }

    return (
        concat_embedder,
        {
            "iter_dict": iter_dict,
            "EXPECTED_EMBEDDING_DIM": EXPECTED_EMBEDDING_DIM,
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_TIME_STEPS": MAX_TIME_STEPS,
        },
    )


class TestConcatEmbedders:
    def test_concat_vanilla_embedders_dim(self, concat_vanilla_embedders):
        embedder, options = concat_vanilla_embedders
        iter_dict = options["iter_dict"]
        batch_size = options["BATCH_SIZE"]
        time_steps = options["MAX_TIME_STEPS"]
        expected_emb_dim = options["EXPECTED_EMBEDDING_DIM"]
        embedding = embedder(iter_dict=iter_dict)
        assert embedding.size() == (batch_size, time_steps, expected_emb_dim)

    def test_concat_vanilla_bow_elmo_embedders_dim(
        self, concat_vanilla_bow_elmo_embedders
    ):
        embedder, options = concat_vanilla_bow_elmo_embedders
        iter_dict = options["iter_dict"]
        batch_size = options["BATCH_SIZE"]
        time_steps = options["MAX_TIME_STEPS"]
        expected_emb_dim = options["EXPECTED_EMBEDDING_DIM"]
        embedding = embedder(iter_dict=iter_dict)
        assert embedding.size() == (batch_size, time_steps, expected_emb_dim)
