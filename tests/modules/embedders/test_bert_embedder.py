import pytest
from sciwing.modules.embedders.bert_embedder import BertEmbedder
import itertools
from sciwing.utils.common import get_system_mem_in_gb
from sciwing.data.line import Line
import torch

bert_types = [
    "bert-base-uncased",
    "bert-base-cased",
    "scibert-base-cased",
    "scibert-sci-cased",
    "scibert-base-uncased",
    "scibert-sci-uncased",
    "bert-large-uncased",
    "bert-large-cased",
]

aggregation_types = ["sum", "average"]

bert_type_aggregation = list(itertools.product(bert_types, aggregation_types))


system_memory = get_system_mem_in_gb()
system_memory = int(system_memory)


@pytest.fixture(params=bert_type_aggregation)
def setup_bert_embedder(request):
    dropout_value = 0.0
    bert_type, aggregation_type = request.param

    bert_embedder = BertEmbedder(
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]

    lines = []
    for string in strings:
        line = Line(text=string)
        lines.append(line)

    return bert_embedder, lines


@pytest.mark.skipif(
    system_memory < 4, reason="System memory too small to run testing for BertEmbedder"
)
class TestBertEmbedder:
    def test_embedder_dimensions(self, setup_bert_embedder):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bert_embedder, lines = setup_bert_embedder
        encoding = bert_embedder(lines)
        lens = [len(line.tokens["tokens"]) for line in lines]
        max_word_len = max(lens)
        assert encoding.size(0) == 2
        assert encoding.size(2) == bert_embedder.get_embedding_dimension()
        assert encoding.size(1) == max_word_len

    def test_bert_embedder_tokens(self, setup_bert_embedder):
        bert_embedder, lines = setup_bert_embedder
        _ = bert_embedder(lines)
        emb_dim = bert_embedder.get_embedding_dimension()
        emb_name = bert_embedder.embedder_name
        for line in lines:
            tokens = line.tokens[bert_embedder.word_tokens_namespace]
            for token in tokens:
                assert isinstance(token.get_embedding(emb_name), torch.FloatTensor)
                assert token.get_embedding(emb_name).size(0) == emb_dim
