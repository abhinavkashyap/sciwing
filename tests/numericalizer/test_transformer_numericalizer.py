import pytest
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizer.transformer_numericalizer import NumericalizerForTransformer


@pytest.fixture
def instances():
    return ["I  like transformers.", "I like python libraries."]


@pytest.fixture(
    params=[
        "bert-base-uncased",
        "bert-base-cased",
        "scibert-base-uncased",
        "scibert-base-cased",
    ]
)
def numericalizer(instances, request):
    bert_type = request.param
    tokenizer = TokenizerForBert(bert_type=bert_type)
    numericalizer = NumericalizerForTransformer(tokenizer=tokenizer)
    return numericalizer


class TestNumericalizeForTransformer:
    def test_token_types(self, numericalizer, instances):
        tokenizer = numericalizer.tokenizer
        for instance in instances:
            tokens = tokenizer.tokenize(instance)
            ids = numericalizer.numericalize_instance(tokens)
            assert all([isinstance(token_id, int) for token_id in ids])

    @pytest.mark.parametrize("padding_length", [10, 100])
    def test_padding_length(self, numericalizer, instances, padding_length):
        tokenizer = numericalizer.tokenizer
        for instance in instances:
            tokens = tokenizer.tokenize(instance)
            ids = numericalizer.numericalize_instance(tokens)
            padded_ids = numericalizer.pad_instance(
                numericalized_text=ids, max_length=padding_length
            )
            assert len(padded_ids) == padding_length
