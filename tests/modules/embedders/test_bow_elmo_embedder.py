import pytest
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.utils.common import pack_to_length
from sciwing.utils.common import get_system_mem_in_gb

mem_gb = get_system_mem_in_gb()
mem_gb = int(mem_gb)


@pytest.mark.skipif(mem_gb < 16, reason="system memory is too low to run elmo embedder")
@pytest.fixture(params=["sum", "average", "first", "last"])
def setup_bow_elmo_encoder(request):
    layer_aggregation = request.param
    instances = ["I like to eat carrot", "I like to go out on long drives in a car"]
    padded_instances = []
    for instance in instances:
        padded_inst = pack_to_length(tokenized_text=instance.split(), max_length=10)
        padded_instances.append(" ".join(padded_inst))
    iter_dict = {"instance": padded_instances}
    bow_elmo_embedder = BowElmoEmbedder(layer_aggregation=layer_aggregation)
    return bow_elmo_embedder, iter_dict


class TestBowElmoEncoder:
    def test_dimension(self, setup_bow_elmo_encoder):
        bow_elmo_embedder, iter_dict = setup_bow_elmo_encoder
        embedding = bow_elmo_embedder(iter_dict)
        assert embedding.size() == (2, 10, 1024)
