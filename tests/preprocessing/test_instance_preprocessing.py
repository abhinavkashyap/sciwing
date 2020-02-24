from sciwing.preprocessing.instance_preprocessing import InstancePreprocessing
import pytest


@pytest.fixture()
def setup_lowercase_tests():
    string = "I LIKE TO MAKE THIS INTO LoWER CASE"
    instance = string.split(" ")
    instance_preprocessing = InstancePreprocessing()
    return string, instance, instance_preprocessing


@pytest.fixture()
def setup_stopwords_test():
    string = "a a but but but"
    return string


@pytest.fixture(
    scope="session",
    params=[
        (
            "lower UPPER Capitalize 123 123a".split(),
            "[LOWER] [UPPER] [CAPITALIZED] [NUMERIC] [ALNUM]".split(),
        ),
        ("UPPER 123-234".split(), "[UPPER] [OTHER]".split()),
        ("(3) 123-234".split(), "[OTHER] [OTHER]".split()),
    ],
)
def setup_tests_indicate_capitalization(request):
    instance = request.param[0]
    expected_instance = request.param[1]
    instance_preprocessing = InstancePreprocessing()
    return instance, expected_instance, instance_preprocessing


class TestInstancePreprocessing:
    def test_lower_case_length(self, setup_lowercase_tests):
        string, instance, instance_preprocessing = setup_lowercase_tests
        lowercased_instance = instance_preprocessing.lowercase(instance)

        assert len(lowercased_instance) == len(instance)

    def test_lower_case(self, setup_lowercase_tests):
        string, instance, instance_preprocessing = setup_lowercase_tests
        lowercased_instance = instance_preprocessing.lowercase(instance)

        for token in lowercased_instance:
            assert token.islower()

    def test_remove_stop_words(self, setup_stopwords_test):
        string = setup_stopwords_test
        instance = string.split()
        instance_preprocess = InstancePreprocessing()

        clean_instance = instance_preprocess.remove_stop_words(instance)
        assert len(clean_instance) == 0

    def test_is_capitalized(self, setup_tests_indicate_capitalization):
        (
            instance,
            expected_instance,
            instance_preprocessing,
        ) = setup_tests_indicate_capitalization
        processed_instance = instance_preprocessing.indicate_capitalization(instance)
        assert len(instance) == len(processed_instance)
        assert processed_instance == expected_instance
