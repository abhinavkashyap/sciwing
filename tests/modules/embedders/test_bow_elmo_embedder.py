import pytest
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.data.line import Line
from sciwing.utils.common import get_system_mem_in_gb
import torch

mem_gb = get_system_mem_in_gb()
mem_gb = int(mem_gb)


@pytest.fixture(params=["sum", "average", "first", "last"])
def setup_bow_elmo_encoder(request):
    layer_aggregation = request.param
    strings = ["I like to eat carrot", "I like to go out on long drives in a car"]

    lines = []
    for string in strings:
        line = Line(text=string)
        lines.append(line)

    bow_elmo_embedder = BowElmoEmbedder(layer_aggregation=layer_aggregation)
    return bow_elmo_embedder, lines


class TestBowElmoEncoder:
    @pytest.mark.slow
    def test_dimension(self, setup_bow_elmo_encoder):
        bow_elmo_embedder, lines = setup_bow_elmo_encoder
        embedding = bow_elmo_embedder(lines)
        assert embedding.size(0) == 2
        assert embedding.size(2) == 1024

    @pytest.mark.slow
    def test_token_embeddings(self, setup_bow_elmo_encoder):
        bow_elmo_embedder, lines = setup_bow_elmo_encoder
        _ = bow_elmo_embedder(lines)

        for line in lines:
            tokens = line.tokens["tokens"]
            for token in tokens:
                assert isinstance(token.get_embedding("elmo"), torch.FloatTensor)
                assert token.get_embedding("elmo").size(0) == 1024
