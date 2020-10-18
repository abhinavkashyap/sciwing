from sciwing.data.seq_sentence import SeqSentence
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
import pytest


class TestSeqSentence:
    def test_sents_word_tokenizers(self):
        sents = ["Nice people", "Great weather"]
        sent = SeqSentence(sents=sents, tokenizers={"tokens": WordTokenizer()})
        tokens = sent.tokens
        assert [[token.text for token in sent_tokens] for sent_tokens in tokens["tokens"]] == [
            ["Nice", "people"],
            ["Great", "weather"]
        ]

    def test_sents_char_tokenizer(self):
        sents = ["Hello", "World"]
        sent = SeqSentence(
            sents=sents,
            tokenizers={"tokens": WordTokenizer(), "chars": CharacterTokenizer()},
        )
        tokens = sent.tokens
        word_tokens = tokens["tokens"]
        char_tokens = tokens["chars"]

        word_tokens = [[tok.text for tok in sent_word_tokens] for sent_word_tokens in word_tokens]
        char_tokens = [[tok.text for tok in sent_char_tokens] for sent_char_tokens in char_tokens]

        assert word_tokens == [["Hello"], ["World"]]
        assert char_tokens == [["H", "e", "l", "l", "o"], ["W", "o", "r", "l", "d"]]

    def test_line_namespaces(self):
        sents = ["Nice people", "Great weather"]
        sent = SeqSentence(sents=sents, tokenizers={"tokens": WordTokenizer()})
        assert sent.namespaces == ["tokens"]
