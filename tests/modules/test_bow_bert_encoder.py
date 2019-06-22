import pytest
from parsect.modules.bow_bert_encoder import BowBertEncoder
import itertools

bert_base_types = [
    "bert-base-uncased",
    "bert-base-cased",
    "scibert-base-cased",
    "scibert-sci-cased",
    "scibert-base-uncased",
    "scibert-sci-uncased",
]

bert_large_types = ["bert-large-uncased", "bert-large-cased"]

aggregation_types = ["sum", "average"]

bert_base_type_agg_type = list(itertools.product(bert_base_types, aggregation_types))
bert_large_type_agg_type = list(itertools.product(bert_large_types, aggregation_types))


@pytest.fixture(scope="module", params=bert_base_type_agg_type)
def setup_bow_bert_encoder_base_type(request):
    emb_dim = 768
    dropout_value = 0.0

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=request.param[1],
        bert_type=request.param[0],
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture(scope="module", params=bert_large_type_agg_type)
def setup_bow_bert_encoder_large_type(request):
    emb_dim = 1024
    dropout_value = 0.0

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=request.param[1],
        bert_type=request.param[0],
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


class TestBowBertEncoder:
    def test_bow_bert_encoder_base_type(self, setup_bow_bert_encoder_base_type):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bow_bert_encoder, strings = setup_bow_bert_encoder_base_type
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 768)

    def test_bow_bert_encoder_large_type(self, setup_bow_bert_encoder_large_type):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bow_bert_encoder, strings = setup_bow_bert_encoder_large_type
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 1024)
