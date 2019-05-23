import pytest
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.vocab.vocab import Vocab


@pytest.fixture
def instances():
    single_instance = [["i", "like", "nlp", "so", "much"]]

    return {"single_instance": single_instance}


@pytest.fixture()
def single_instance_less_tokens_setup(instances):
    single_instance = instances["single_instance"]
    MAX_LENGTH = 10
    MAX_NUM_WORDS = 100

    vocabulary = Vocab(instances=single_instance, max_num_words=MAX_NUM_WORDS)

    numericalizer = Numericalizer(max_length=MAX_LENGTH, vocabulary=vocabulary)

    return single_instance, MAX_LENGTH, numericalizer, vocabulary


@pytest.fixture()
def single_instance_more_tokens_setup(instances):
    single_instance = instances["single_instance"]
    MAX_LENGTH = 3
    MAX_NUM_WORDS = 100

    vocabulary = Vocab(instances=single_instance, max_num_words=MAX_NUM_WORDS)

    numericalizer = Numericalizer(max_length=MAX_LENGTH, vocabulary=vocabulary)

    return single_instance, MAX_LENGTH, numericalizer, vocabulary


@pytest.fixture()
def single_instance_2_tokens_setup(instances):
    single_instance = instances["single_instance"]
    MAX_LENGTH = 2
    MAX_NUM_WORDS = 100

    vocabulary = Vocab(instances=single_instance, max_num_words=MAX_NUM_WORDS)

    numericalizer = Numericalizer(max_length=MAX_LENGTH, vocabulary=vocabulary)

    return single_instance, MAX_LENGTH, numericalizer, vocabulary


class TestNumericalizer:
    def test_max_length_instance_has_less_tokens_than_max_length(
        self, single_instance_less_tokens_setup
    ):
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_less_tokens_setup
        )

        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        assert len(numerical_tokens) == MAX_LENGTH

    def test_start_end_token_instance_has_less_tokens(
        self, single_instance_less_tokens_setup
    ):
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_less_tokens_setup
        )
        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        assert numerical_tokens[0] == vocabulary.get_idx_from_token(
            vocabulary.start_token
        )
        assert numerical_tokens[-1] == vocabulary.get_idx_from_token(
            vocabulary.pad_token
        )

    def test_max_length_instance_has_more_tokens_than_max_length(
        self, single_instance_more_tokens_setup
    ):
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_more_tokens_setup
        )
        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        assert len(numerical_tokens) == MAX_LENGTH

    def test_start_end_token_instance_has_more_tokens_than_max_length(
        self, single_instance_more_tokens_setup
    ):
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_more_tokens_setup
        )
        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        assert numerical_tokens[0] == vocabulary.get_idx_from_token(
            vocabulary.start_token
        )
        assert numerical_tokens[-1] == vocabulary.get_idx_from_token(
            vocabulary.end_token
        )

    def test_max_lengths_instance_has_less_tokens_than_max_length(
        self, single_instance_less_tokens_setup
    ):
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_less_tokens_setup
        )

        len_tokens, numerical_tokens = numericalizer.numericalize_batch_instances(
            [single_instance[0], single_instance[0]]
        )

        assert len(numerical_tokens[0]) == MAX_LENGTH
        assert len(numerical_tokens[1]) == MAX_LENGTH
        assert len_tokens[0] == len(single_instance[0])
        assert len_tokens[1] == len(single_instance[0])

    def test_start_end_token_max_length_2(self, single_instance_2_tokens_setup):
        """
            IF THE MAX LENGTH IS 2, THEN THE FIRST TOKEN SHOULD BE START
            THE SECOND TOKEN SHOULD BE END OF THE SENTENCE
        """
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_2_tokens_setup
        )
        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        assert numerical_tokens[0] == vocabulary.get_idx_from_token(
            vocabulary.start_token
        )
        assert numerical_tokens[1] == vocabulary.get_idx_from_token(
            vocabulary.end_token
        )

    def test_tokens_are_integers(self, single_instance_less_tokens_setup):
        """
        Just to test that something untoward is happening
        """
        single_instance, MAX_LENGTH, numericalizer, vocabulary = (
            single_instance_less_tokens_setup
        )
        len_tokens, numerical_tokens = numericalizer.numericalize_instance(
            single_instance[0]
        )

        for each_token in numerical_tokens:
            assert type(each_token) == int
