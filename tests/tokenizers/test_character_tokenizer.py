import pytest
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer


@pytest.fixture
def setup_character_tokenizer():
    tokenizer = CharacterTokenizer()
    return tokenizer


class TestCharacterTokenizer:
    @pytest.mark.parametrize(
        "string, expected_len", [("The", 3), ("The quick brown", 15)]
    )
    def test_character_tokenizer_length(
        self, string, expected_len, setup_character_tokenizer
    ):
        char_tokenizer = setup_character_tokenizer
        tokenized = char_tokenizer.tokenize(string)
        assert len(tokenized) == expected_len

    @pytest.mark.parametrize(
        "string, expected_tokenization",
        [
            ("The", ["T", "h", "e"]),
            (
                "The quick @#",
                ["T", "h", "e", " ", "q", "u", "i", "c", "k", " ", "@", "#"],
            ),
        ],
    )
    def test_character_tokenizer(
        self, string, expected_tokenization, setup_character_tokenizer
    ):
        tokenizer = setup_character_tokenizer
        tokenized = tokenizer.tokenize(string)
        assert tokenized == expected_tokenization

    @pytest.mark.parametrize("batch, expected_len", [(["The", "The quick brown"], 2)])
    def test_batch_tokenization_len(
        self, batch, expected_len, setup_character_tokenizer
    ):
        tokenizer = setup_character_tokenizer
        tokenized = tokenizer.tokenize_batch(batch)
        assert len(tokenized) == 2
