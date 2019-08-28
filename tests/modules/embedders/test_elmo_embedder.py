import pytest
from sciwing.modules.embedders.elmo_embedder import ElmoEmbedder


@pytest.fixture
def setup_elmo_embedder():
    elmo_embedder = ElmoEmbedder()
    instances = ["I like to test elmo".split(), "Elmo context embedder".split()]
    return elmo_embedder, instances


class TestElmoEmbedder:
    def test_elmo_embedder_dimensions(self, setup_elmo_embedder):
        elmo_embedder, instances = setup_elmo_embedder
        embedding = elmo_embedder(instances)
        assert embedding.size() == (len(instances), 5, 1024)
