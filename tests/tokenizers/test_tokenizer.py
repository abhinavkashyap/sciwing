from sciwing.tokenizers.word_tokenizer import WordTokenizer
import pytest


class TestWordTokenizer:
    def test_sample_word_tokenization(self):
        sample_sentence = "I like big apple."
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(sample_sentence)

        assert tokens == ["I", "like", "big", "apple", "."]

    def test_sample_apostrophe_tokenization(self):
        sample_sentence = "I don't like apples."
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(sample_sentence)

        assert tokens == ["I", "do", "n't", "like", "apples", "."]

    def test_len_sample_batch(self):
        sample_sentences = ["I like big apple.", "We process text"]
        tokenizer = WordTokenizer()
        tokenized = tokenizer.tokenize_batch(sample_sentences)
        assert len(tokenized) == 2

    def test_word_tokenization_types(self):
        with pytest.raises(AssertionError):
            tokenizer = WordTokenizer(tokenizer="moses")

    # TODO: Remove this after you have implemented nltk tokenization
    def test_other_tokenizer(self):
        tokenizer = WordTokenizer(tokenizer="nltk")
        assert tokenizer.tokenize("First string") is None

    def test_vanilla_tokenizer(self):
        tokenizer = WordTokenizer(tokenizer="vanilla")
        tokenized = tokenizer.tokenize(
            "(1999). & P., W. The Control of Discrete Event Systems."
        )
        assert tokenized == [
            "(1999).",
            "&",
            "P.,",
            "W.",
            "The",
            "Control",
            "of",
            "Discrete",
            "Event",
            "Systems.",
        ]

    def test_spacy_whitespace_tokenizer(self):
        tokenizer = WordTokenizer(tokenizer="spacy-whitespace")
        tokenized = tokenizer.tokenize(
            "(1999). & P., W. The Control of Discrete Event Systems."
        )
        assert tokenized == [
            "(1999).",
            "&",
            "P.,",
            "W.",
            "The",
            "Control",
            "of",
            "Discrete",
            "Event",
            "Systems.",
        ]
