import pytest
from parsect.modules.bow_elmo_encoder import BowElmoEncoder


@pytest.fixture
def setup_bow_elmo_encoder_agg_sum():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]
    bow_elmo_encoder = BowElmoEncoder(aggregation_type="sum")
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_agg_average():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]
    bow_elmo_encoder = BowElmoEncoder(aggregation_type="average")
    return bow_elmo_encoder, instances


class TestBowElmoEncoder:
    def test_bow_elmo_encoder_dimension_sum(self, setup_bow_elmo_encoder_agg_sum):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_agg_sum
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_bow_elmo_encoder_dimension_average(
        self, setup_bow_elmo_encoder_agg_average
    ):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_agg_average
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)
