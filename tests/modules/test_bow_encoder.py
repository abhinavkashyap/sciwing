import numpy as np
import pytest
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery


@pytest.fixture(params=["sum", "average"])
def setup_bow_encoder(request):
    aggregation_type = request.param
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    encoder = BOW_Encoder(embedder=embedder, aggregation_type=aggregation_type)
    texts = ["First sentence", "second sentence"]
    lines = []
    for text in texts:
        line = Line(text=text)
        lines.append(line)

    return encoder, lines


class TestBOWEncoder:
    def test_bow_encoder_dimensions(self, setup_bow_encoder):
        encoder, lines = setup_bow_encoder
        encoded_lines = encoder(lines)
        assert encoded_lines.size(0) == 2
        assert encoded_lines.size(1) == 50

    def test_bow_encoder_in_class_nursery(self):
        assert ClassNursery.class_nursery.get("BOW_Encoder") is not None
