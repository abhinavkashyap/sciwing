import pytest
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery
import torch
from sciwing.utils.common import get_system_mem_in_gb

mem_in_gb = get_system_mem_in_gb()


@pytest.fixture(
    params=["glove_6B_50", "glove_6B_100", "glove_6B_200", "glove_6B_300", "parscit"]
)
def setup_embedder(request):
    embedding_type = request.param
    embedder = WordEmbedder(embedding_type)
    return embedder


@pytest.fixture
def setup_lines():
    texts = ["first line", "second line"]
    lines = []
    for text in texts:
        line = Line(text=text)
        lines.append(line)
    return lines


@pytest.mark.skipif(
    int(mem_in_gb) < 4, reason="Memory is too low to run bert tokenizers"
)
class TestWordEmbedder:
    def test_dimension(self, setup_embedder, setup_lines):
        embedder = setup_embedder
        lines = setup_lines
        _ = embedder(lines)
        for line in lines:
            for token in line.tokens["tokens"]:
                embedding = token.get_embedding(name=embedder.embedding_type)
                assert isinstance(embedding, torch.FloatTensor)

    def test_final_embedding_size(self, setup_embedder, setup_lines):
        embedder = setup_embedder
        lines = setup_lines
        embeddings = embedder(lines)
        assert embeddings.size(0) == 2

    def test_vanilla_embedder_in_class_nursery(self):
        assert ClassNursery.class_nursery["WordEmbedder"] is not None
