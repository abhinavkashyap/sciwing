import pytest
from sciwing.modules.embedders.bert_embedder import BertEmbedder
import itertools
from sciwing.utils.common import get_system_mem_in_gb

bert_base_types = [
    "bert-base-uncased",
    "bert-base-cased",
    "scibert-base-cased",
    "scibert-sci-cased",
    "scibert-base-uncased",
    "scibert-sci-uncased",
]

bert_large_types = ["bert-large-uncased", "bert-large-cased"]

aggregation_types = ["sum", "average"]

bert_base_type_agg_type = list(itertools.product(bert_base_types, aggregation_types))
bert_large_type_agg_type = list(itertools.product(bert_large_types, aggregation_types))

system_memory = get_system_mem_in_gb()
system_memory = int(system_memory)


@pytest.fixture(scope="module", params=bert_base_type_agg_type)
def setup_bert_embedder(request):
    emb_dim = 768
    dropout_value = 0.0

    bow_bert_encoder = BertEmbedder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=request.param[1],
        bert_type=request.param[0],
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]
    iter_dict = {"raw_instance": strings}

    return bow_bert_encoder, iter_dict


@pytest.fixture(scope="module", params=bert_large_type_agg_type)
def setup_bert_embedder_large(request):
    emb_dim = 1024
    dropout_value = 0.0

    bow_bert_encoder = BertEmbedder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=request.param[1],
        bert_type=request.param[0],
    )
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]
    iter_dict = {"raw_instance": strings}
    return bow_bert_encoder, iter_dict


@pytest.mark.skipif(
    system_memory < 20, reason="System memory too small to run BERT Embedder"
)
class TestBertEmbedder:
    def test_bert_embedder_base_type(self, setup_bert_embedder):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bert_embedder, iter_dict = setup_bert_embedder
        encoding = bert_embedder(iter_dict)
        assert encoding.size() == (2, 8, 768)

    def test_bert_embedder_encoder_large_type(self, setup_bert_embedder_large):
        """
            The bow bert encoder should return a single instance
            that is the sum of the word embeddings of the instance
        """
        bert_embedder, iter_dict = setup_bert_embedder_large
        encoding = bert_embedder(iter_dict)
        assert encoding.size() == (2, 8, 1024)
