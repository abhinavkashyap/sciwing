import pytest
from sciwing.data.line import Line
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer


class TestLine:
    def test_line_word_tokenizers(self):
        text = "This is a single line"
        line = Line(text=text, tokenizers={"tokens": WordTokenizer()})
        tokens = line.tokens
        assert [token.text for token in tokens["tokens"]] == [
            "This",
            "is",
            "a",
            "single",
            "line",
        ]

    def test_line_char_tokenizer(self):
        text = "Word"
        line = Line(
            text=text,
            tokenizers={"tokens": WordTokenizer(), "chars": CharacterTokenizer()},
        )
        tokens = line.tokens
        word_tokens = tokens["tokens"]
        char_tokens = tokens["chars"]

        word_tokens = [tok.text for tok in word_tokens]
        char_tokens = [tok.text for tok in char_tokens]

        assert word_tokens == ["Word"]
        assert char_tokens == ["W", "o", "r", "d"]

    def test_line_namespaces(self):
        text = "Single line"
        line = Line(text=text, tokenizers={"tokens": WordTokenizer()})
        assert line.namespaces == ["tokens"]
