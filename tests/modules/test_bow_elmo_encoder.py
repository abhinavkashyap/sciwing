import pytest
from parsect.modules.bow_elmo_encoder import BowElmoEncoder
from parsect.utils.common import pack_to_length


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_sum_word_agg_sum():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]
    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(layer_aggregation="sum", word_aggregation="sum")
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_average_word_agg_sum():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]
    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(
        layer_aggregation="average", word_aggregation="sum"
    )
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_last_word_agg_sum():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(layer_aggregation="last", word_aggregation="sum")
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_first_word_agg_sum():
    instances = [
        "I like to eat carrots".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(layer_aggregation="first", word_aggregation="sum")
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_sum_word_agg_average():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(
        layer_aggregation="sum", word_aggregation="average"
    )
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_average_word_agg_average():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(
        layer_aggregation="average", word_aggregation="average"
    )
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_last_word_agg_average():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(
        layer_aggregation="last", word_aggregation="average"
    )
    return bow_elmo_encoder, instances


@pytest.fixture
def setup_bow_elmo_encoder_layer_agg_first_word_agg_average():
    instances = [
        "I like to eat carrot".split(),
        "I like to go out on long drives in a car".split(),
    ]

    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance, max_length=10)
        padded_instances.append(padded_inst)
    instances = padded_instances

    bow_elmo_encoder = BowElmoEncoder(
        layer_aggregation="first", word_aggregation="average"
    )
    return bow_elmo_encoder, instances


class TestBowElmoEncoder:
    def test_dimension_agg_sum_word_agg_sum(
        self, setup_bow_elmo_encoder_layer_agg_sum_word_agg_sum
    ):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_layer_agg_sum_word_agg_sum
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_average_word_agg_sum(
        self, setup_bow_elmo_encoder_layer_agg_average_word_agg_sum
    ):
        bow_elmo_encoder, instances = (
            setup_bow_elmo_encoder_layer_agg_average_word_agg_sum
        )
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_last_layer_word_agg_sum(
        self, setup_bow_elmo_encoder_layer_agg_last_word_agg_sum
    ):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_layer_agg_last_word_agg_sum
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_first_layer_word_agg_sum(
        self, setup_bow_elmo_encoder_layer_agg_first_word_agg_sum
    ):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_layer_agg_first_word_agg_sum
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_sum_word_agg_average(
        self, setup_bow_elmo_encoder_layer_agg_sum_word_agg_average
    ):
        bow_elmo_encoder, instances = (
            setup_bow_elmo_encoder_layer_agg_sum_word_agg_average
        )
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_average_word_agg_average(
        self, setup_bow_elmo_encoder_layer_agg_average_word_agg_average
    ):
        bow_elmo_encoder, instances = (
            setup_bow_elmo_encoder_layer_agg_average_word_agg_average
        )
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_last_word_agg_average(
            self, setup_bow_elmo_encoder_layer_agg_last_word_agg_average
    ):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_layer_agg_last_word_agg_average
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

    def test_dimension_agg_first_word_agg_average(
            self, setup_bow_elmo_encoder_layer_agg_first_word_agg_average):
        bow_elmo_encoder, instances = setup_bow_elmo_encoder_layer_agg_first_word_agg_average
        len_instances = len(instances)
        embedding = bow_elmo_encoder(instances)
        assert embedding.size() == (len_instances, 1024)

