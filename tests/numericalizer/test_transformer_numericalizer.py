import pytest
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizers.transformer_numericalizer import NumericalizerForTransformer
import torch
from sciwing.utils.common import get_system_mem_in_gb


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


mem_in_gb = get_system_mem_in_gb()


@pytest.mark.skipif(
    int(mem_in_gb) < 10, reason="Memory is too low to run bert tokenizers"
)
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

    def test_get_mask(self, numericalizer, instances):
        tokenizer = numericalizer.tokenizer
        max_length = 10
        for instance in instances:
            tokens = tokenizer.tokenize(instance)
            len_tokens = len(tokens)
            ids = numericalizer.numericalize_instance(tokens)
            padded_ids = numericalizer.pad_instance(
                numericalized_text=ids, max_length=max_length, add_start_end_token=False
            )
            padding_length = max_length - len_tokens

            mask = [0] * len_tokens + [1] * padding_length
            expected_mask = torch.ByteTensor(mask)
            mask = numericalizer.get_mask_for_instance(instance=padded_ids)
            assert torch.all(torch.eq(expected_mask, mask))
