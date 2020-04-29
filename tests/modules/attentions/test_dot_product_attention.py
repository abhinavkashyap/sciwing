import pytest
import torch
from sciwing.modules.attentions.dot_product_attention import DotProductAttention

N = 1
H = 100
T = 10


@pytest.fixture()
def zero_query():
    query = torch.zeros(N, H)
    return query


@pytest.fixture()
def random_keys():
    key = torch.randn(N, T, H)
    return key


@pytest.fixture
def attention():
    attention = DotProductAttention()
    return attention


class TestDotProductAttention:
    def test_zero_query(self, zero_query, random_keys, attention):
        """
        Zero query should give uninform distribution over the keys
        """
        attentions = attention(query_matrix=zero_query, key_matrix=random_keys)
        N, T = attentions.size()
        ones = torch.ones(N, T)
        ones = ones / T
        assert torch.allclose(attentions, ones)

    def test_attention_sums_to_one(sel, zero_query, random_keys, attention):
        attentions = attention(query_matrix=zero_query, key_matrix=random_keys)
        N, T = attentions.size()
        ones = torch.ones(N, 1)
        sum = torch.sum(attentions, dim=1)
        assert torch.allclose(ones, sum)

    def test_attention_size(self, zero_query, random_keys, attention):
        attentions = attention(query_matrix=zero_query, key_matrix=random_keys)
        assert attentions.size() == (N, T)
