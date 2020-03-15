import pytest
from sciwing.modules.embedders.flair_embedder import FlairEmbedder
from sciwing.data.line import Line
from sciwing.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture
def flair_embedder():
    embedder = FlairEmbedder(embedding_type="news", datasets_manager=None)

    return embedder


@pytest.fixture
def lines():
    texts = ["First line", "Second Line which is longer"]
    lines = []
    for text in texts:
        line = Line(
            text=text, tokenizers={"tokens": WordTokenizer(tokenizer="vanilla")}
        )
        lines.append(line)

    return lines


class TestFlairEmbedder:
    def test_embedding_dimension(self, flair_embedder, lines):
        embedding = flair_embedder(lines)
        assert embedding.dim() == 3
