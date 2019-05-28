import pytest
from parsect.modules.bow_bert_encoder import BowBertEncoder


@pytest.fixture
def setup_bow_bert_encoder_basic_uncased_sum():
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "sum"
    bert_type = "bert-base-uncased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_basic_uncased_average():
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "average"
    bert_type = "bert-base-uncased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_basic_cased_sum():
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "sum"
    bert_type = "bert-base-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_basic_cased_average():
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "average"
    bert_type = "bert-base-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_large_uncased_sum():
    emb_dim = 1024
    dropout_value = 0.0
    aggregation_type = "sum"
    bert_type = "bert-large-uncased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_large_uncased_average():
    emb_dim = 1024
    dropout_value = 0.0
    aggregation_type = "average"
    bert_type = "bert-large-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_large_cased_sum():
    emb_dim = 1024
    dropout_value = 0.0
    aggregation_type = "sum"
    bert_type = "bert-large-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


@pytest.fixture
def setup_bow_bert_encoder_large_cased_average():
    emb_dim = 1024
    dropout_value = 0.0
    aggregation_type = "average"
    bert_type = "bert-large-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    return bow_bert_encoder, strings


class TestBowBertEncoder:
    def test_base_uncased_sum(self, setup_bow_bert_encoder_basic_uncased_sum):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bow_bert_encoder, strings = setup_bow_bert_encoder_basic_uncased_sum
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 768)

    def test_base_uncased_average(self, setup_bow_bert_encoder_basic_uncased_average):
        bow_bert_encoder, strings = setup_bow_bert_encoder_basic_uncased_average
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 768)

    def test_base_cased_sum(self, setup_bow_bert_encoder_basic_cased_sum):
        bow_bert_encoder, strings = setup_bow_bert_encoder_basic_cased_sum
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 768)

    def test_base_cased_average(self, setup_bow_bert_encoder_basic_cased_average):
        bow_bert_encoder, strings = setup_bow_bert_encoder_basic_cased_average
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 768)

    def test_large_uncased_sum(self, setup_bow_bert_encoder_large_uncased_sum):
        bow_bert_encoder, strings = setup_bow_bert_encoder_large_uncased_sum
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 1024)

    def test_large_uncased_average(self, setup_bow_bert_encoder_large_uncased_average):
        bow_bert_encoder, strings = setup_bow_bert_encoder_large_uncased_average
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 1024)

    def test_large_cased_sum(self, setup_bow_bert_encoder_large_cased_sum):
        bow_bert_encoder, strings = setup_bow_bert_encoder_large_cased_sum
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 1024)

    def test_large_cased_average(self, setup_bow_bert_encoder_large_cased_average):
        bow_bert_encoder, strings = setup_bow_bert_encoder_large_cased_average
        encoding = bow_bert_encoder(strings)
        assert encoding.size() == (2, 1024)
