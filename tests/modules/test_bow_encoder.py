import numpy as np
import pytest
import torch
import torch.nn as nn
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
import itertools
from parsect.utils.class_nursery import ClassNursery

aggregation_types = ["sum", "average"]
embedding_type = ["zeros", "ones"]

params = itertools.product(aggregation_types, embedding_type)


@pytest.fixture(params=params)
def setup_zero_embeddings(request):
    EMB_DIM = 300
    VOCAB_SIZE = 10
    BATCH_SIZE = 10
    aggregation_type = request.param[0]
    embedding_type = request.param[1]
    embedding = None
    if embedding_type == "zeros":
        embedding = torch.zeros([VOCAB_SIZE, EMB_DIM])
    elif embedding_type == "ones":
        embedding = torch.ones([VOCAB_SIZE, EMB_DIM])
    embedding = nn.Embedding.from_pretrained(embedding)
    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=EMB_DIM)
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM, embedder=embedder, aggregation_type=aggregation_type
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, EMB_DIM))
    tokens = torch.LongTensor(tokens)
    iter_dict = {"tokens": tokens}
    options = {
        "EMB_DIM": EMB_DIM,
        "VOCAB_SIZE": VOCAB_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "EMBEDDING_TYPE": embedding_type,
        "AGGREGATION_TYPE": aggregation_type,
    }
    return encoder, iter_dict, options


class TestBOWEncoder:
    def test_bow_encoder_zeros(self, setup_zero_embeddings):
        encoder, iter_dict, options = setup_zero_embeddings
        BATCH_SIZE = options["BATCH_SIZE"]
        EMB_DIM = options["EMB_DIM"]
        EMBEDDING_TYPE = options["EMBEDDING_TYPE"]
        AGGREGATION_TYPE = options["AGGREGATION_TYPE"]
        embeddings = encoder(iter_dict=iter_dict)
        if EMBEDDING_TYPE == "zeros" and AGGREGATION_TYPE == "average":
            expected_embedding = torch.zeros([BATCH_SIZE, EMB_DIM])
        elif EMBEDDING_TYPE == "zeros" and AGGREGATION_TYPE == "sum":
            expected_embedding = torch.zeros([BATCH_SIZE, EMB_DIM])
        elif EMBEDDING_TYPE == "ones" and AGGREGATION_TYPE == "average":
            expected_embedding = torch.ones([BATCH_SIZE, EMB_DIM])
        elif EMBEDDING_TYPE == "ones" and AGGREGATION_TYPE == "sum":
            expected_embedding = torch.ones([BATCH_SIZE, EMB_DIM]) * EMB_DIM

        assert embeddings.size() == (BATCH_SIZE, EMB_DIM)
        assert torch.all(torch.eq(embeddings, expected_embedding)).item()

    def test_bow_encoder_in_class_nursery(self):
        assert ClassNursery.class_nursery.get("BOW_Encoder") is not None
