import pytest
from sciwing.modules.embedders.elmo_embedder import ElmoEmbedder
from sciwing.utils.common import get_system_mem_in_gb
from sciwing.data.line import Line

mem_gb = get_system_mem_in_gb()
mem_gb = int(mem_gb)


@pytest.fixture
def setup_elmo_embedder():
    elmo_embedder = ElmoEmbedder()
    texts = ["I like to test elmo", "Elmo context embedder"]
    lines = []
    for text in texts:
        line = Line(text=text)
        lines.append(line)
    return elmo_embedder, lines


@pytest.mark.skipif(
    mem_gb < 16, reason="skipping ELMO embedder because system memory is low"
)
class TestElmoEmbedder:
    def test_elmo_embedder_dimensions(self, setup_elmo_embedder):
        elmo_embedder, lines = setup_elmo_embedder
        embedding = elmo_embedder(lines)
        assert embedding.size() == (len(lines), 5, 1024)
