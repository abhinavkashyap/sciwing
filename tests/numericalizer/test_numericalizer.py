import pytest
from sciwing.numericalizer.numericalizer import Numericalizer
from sciwing.vocab.vocab import Vocab


@pytest.fixture
def instances():
    single_instance = [["i", "like", "nlp", "so", "much"]]

    return {"single_instance": single_instance}


@pytest.fixture()
def single_instance_setup(instances):
    single_instance = instances["single_instance"]
    MAX_NUM_WORDS = 100

    vocabulary = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)

    numericalizer = Numericalizer(vocabulary=vocabulary)

    return single_instance, numericalizer, vocabulary


class TestNumericalizer:
    def test_max_length_instance_has_less_tokens_than_max_length(
        self, single_instance_setup
    ):
        single_instance, numericalizer, vocabulary = single_instance_setup

        numerical_tokens = numericalizer.numericalize_instance(single_instance[0])

        assert len(numerical_tokens) == len(single_instance[0])

    def test_tokens_are_integers(self, single_instance_setup):
        """
        Just to test that something untoward is happening
        """
        single_instance, numericalizer, vocabulary = single_instance_setup
        numerical_tokens = numericalizer.numericalize_instance(single_instance[0])

        for each_token in numerical_tokens:
            assert type(each_token) == int
