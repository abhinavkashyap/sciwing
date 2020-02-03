import pytest
from sciwing.data.token import Token
import numpy as np


class TestToken:
    def test_token_initialization(self):
        token = Token("token")
        assert token.text == "token"

    def test_token_len(self):
        token = Token("token")
        assert token.len == 5

    @pytest.mark.parametrize(
        "embedding_type, embedding",
        [("glove", np.random.rand(100)), ("bert", np.random.rand(1000))],
    )
    def test_token_set_embedding(self, embedding_type, embedding):
        token = Token("token")
        try:
            token.set_embedding(name=embedding_type, value=embedding)
        except:
            pytest.fail("setting the embedding failed")
