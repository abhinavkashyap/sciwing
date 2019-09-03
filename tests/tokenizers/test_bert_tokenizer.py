from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
import pytest
from sciwing.utils.common import get_system_mem_in_gb


@pytest.fixture()
def setup_bert_tokenizer():
    def _setup_bert_tokenizer(bert_type):
        return TokenizerForBert(bert_type)

    return _setup_bert_tokenizer


@pytest.fixture()
def setup_strings():
    return ["BERT and many other language models", "Introduction .", "123 abc frem"]


bert_types = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "scibert-base-cased",
    "scibert-sci-cased",
    "scibert-base-uncased",
    "scibert-sci-uncased",
]

mem_in_gb = get_system_mem_in_gb()


@pytest.mark.skipif(
    int(mem_in_gb) < 16, reason="Memory is too low to run bert tokenizers"
)
class TestTokenizerForBert:
    @pytest.mark.parametrize("bert_type", bert_types)
    def test_tokenizer_initializations(self, bert_type, setup_bert_tokenizer):
        try:
            tokenizer = setup_bert_tokenizer(bert_type)
            assert tokenizer.tokenizer is not None
        except:
            pytest.fail(f"Failed to setup tokenizer for bert type {bert_type}")

    @pytest.mark.parametrize("bert_type", bert_types)
    def test_tokenizer_returns_string_list(
        self, bert_type, setup_bert_tokenizer, setup_strings
    ):
        try:
            tokenizer = setup_bert_tokenizer(bert_type)
            strings = setup_strings
            for string in strings:
                assert type(tokenizer.tokenize(string)) == list
        except:
            pytest.fail(f"Failed to setup tokenizer for bert type {bert_type}")

    @pytest.mark.parametrize("bert_type", bert_types)
    def test_len_tokenization(self, bert_type, setup_bert_tokenizer, setup_strings):
        tokenizer = setup_bert_tokenizer(bert_type)
        strings = setup_strings
        for string in strings:
            assert len(tokenizer.tokenize(string)) > 0

    @pytest.mark.parametrize("bert_type", bert_types)
    def test_sample_word_tokenization(self, bert_type, setup_bert_tokenizer):
        sample_sentence = "I like big apple."
        tokenizer = setup_bert_tokenizer(bert_type)
        tokens = tokenizer.tokenize(sample_sentence)
        tokens = list(map(lambda token: token.lower(), tokens))
        expected_tokens = ["I", "like", "big", "apple", "."]
        expected_tokens = list(map(lambda token: token.lower(), expected_tokens))

        assert tokens == expected_tokens

    @pytest.mark.parametrize("bert_type", bert_types)
    def test_len_sample_batch(self, bert_type, setup_bert_tokenizer):
        sample_sentences = ["I like big apple.", "We process text"]
        tokenizer = setup_bert_tokenizer(bert_type)
        tokenized = tokenizer.tokenize_batch(sample_sentences)
        assert len(tokenized) == 2
