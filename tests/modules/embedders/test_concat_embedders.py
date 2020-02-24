import pytest
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.line import Line
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer


@pytest.fixture
def setup_lines():
    texts = ["First sentence", "Second Sentence"]
    lines = []
    for text in texts:
        line = Line(
            text=text,
            tokenizers={"tokens": WordTokenizer(), "char_tokens": CharacterTokenizer()},
        )
        lines.append(line)
    return lines


@pytest.fixture
def setup_concat_vanilla_embedders():
    word_embedder_1 = WordEmbedder(embedding_type="glove_6B_50")
    word_embedder_2 = WordEmbedder(embedding_type="glove_6B_100")
    embedder = ConcatEmbedders([word_embedder_1, word_embedder_2])
    return embedder


class TestConcatEmbedders:
    def test_concat_vanilla_embedders_dim(
        self, setup_concat_vanilla_embedders, setup_lines
    ):
        concat_embedder = setup_concat_vanilla_embedders
        lines = setup_lines
        embedding = concat_embedder(lines)
        assert embedding.size(0) == 2
        assert embedding.size(2) == 150
