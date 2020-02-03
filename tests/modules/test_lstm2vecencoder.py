import pytest
import torch
import torch.nn as nn
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.line import Line
import itertools

directions = [True, False]  # True for bi-directions
combination_strategy = ["concat", "sum"]

directions_combine_strategy = itertools.product(directions, combination_strategy)
directions_combine_strategy = list(directions_combine_strategy)


@pytest.fixture(params=directions_combine_strategy)
def setup_lstm2vecencoder(request):
    hidden_dimension = 1024
    combine_strategy = request.param[1]
    bidirectional = request.param[0]
    embedder = WordEmbedder(embedding_type="glove_6B_50")

    encoder = LSTM2VecEncoder(
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=hidden_dimension,
        bidirectional=bidirectional,
        combine_strategy=combine_strategy,
        rnn_bias=False,
    )

    texts = ["First sentence", "second sentence"]
    lines = []
    for text in texts:
        line = Line(text=text)
        lines.append(line)

    return (
        encoder,
        {
            "hidden_dim": 2 * hidden_dimension
            if bidirectional and combine_strategy == "concat"
            else hidden_dimension,
            "bidirectional": False,
            "combine_strategy": combine_strategy,
            "lines": lines,
        },
    )


class TestLstm2VecEncoder:
    def test_raises_error_on_wrong_combine_strategy(self, setup_lstm2vecencoder):
        with pytest.raises(AssertionError):

            encoder = LSTM2VecEncoder(
                embedder=WordEmbedder("glove_6B_50"), combine_strategy="add"
            )

    def test_encoding_dimension(self, setup_lstm2vecencoder):
        encoder, options = setup_lstm2vecencoder
        hidden_dim = options["hidden_dim"]
        lines = options["lines"]
        batch_size = len(lines)
        encoding = encoder(lines=lines)
        assert encoding.size() == (batch_size, hidden_dim)
