import pytest
from sciwing.data.token import Token


class TestToken:
    def test_token_initialization(self):
        token = Token("token")
        assert token.text == "token"

    def test_token_len(self):
        token = Token("token")
        assert token.len == 5
