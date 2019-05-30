"""
This tests the pipeline till numericalizatio
"""
import pytest

import parsect.constants as constants
from parsect.utils.common import convert_secthead_to_json
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.preprocessing.instance_preprocessing import InstancePreprocessing
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


# 1. Convert parsect data to json
@pytest.fixture()
def get_parsect_data():
    parsect_json = convert_secthead_to_json(SECT_LABEL_FILE)
    return parsect_json


# 2. Convert the json to instances
@pytest.fixture()
def get_tokenized_data(get_parsect_data):
    parsect_json = get_parsect_data
    parsect_lines = parsect_json["parse_sect"]
    parsect_lines = parsect_lines[:100]
    tokenizer = WordTokenizer()

    lines = []
    labels = []

    for line_json in parsect_lines:
        text = line_json["text"]
        label = line_json["label"]
        lines.append(text)
        labels.append(label)

    instances = tokenizer.tokenize_batch(lines)

    return instances, labels


# 3. Perform pre-processing on instances
@pytest.fixture()
def get_preprocessed_instances(get_tokenized_data):
    instances, labels = get_tokenized_data
    instance_preprocessing = InstancePreprocessing()
    instances = list(map(instance_preprocessing.lowercase, instances))
    return instances, labels


# 4. Numericalization of tokens
@pytest.fixture()
def get_numericalized_instances(get_preprocessed_instances):
    instances, labels = get_preprocessed_instances
    MAX_NUM_WORDS = 3000
    MAX_LENGTH = 15
    vocab = Vocab(instances, max_num_words=MAX_NUM_WORDS)
    vocab.build_vocab()
    numericalizer = Numericalizer(vocabulary=vocab)
    numericalized_instances = numericalizer.numericalize_batch_instances(instances[:32])
    return {
        "numericalized_instances": numericalized_instances,
        "labels": labels,
        "max_length": MAX_LENGTH,
        "max_num_words": MAX_NUM_WORDS,
        "vocab": vocab,
    }


class TestPipeline:
    def test_integers(self, get_numericalized_instances):
        numericalized_instances = get_numericalized_instances["numericalized_instances"]
        for instance in numericalized_instances:
            assert all([type(token) == int for token in instance])

    def test_max_vocab(self, get_numericalized_instances):
        numericalized_instances = get_numericalized_instances["numericalized_instances"]
        vocab = get_numericalized_instances["vocab"]
        MAX_NUM_WORDS = get_numericalized_instances["max_num_words"]
        vocab_len = vocab.get_vocab_len()
        assert vocab_len <= MAX_NUM_WORDS + len(vocab.special_vocab)
