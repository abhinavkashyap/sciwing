import pytest
from sciwing.modules.lstm2seqdecoder import Lstm2SeqDecoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import AbstractiveSummarizationDatasetManager
from sciwing.data.line import Line
import itertools
import torch

lstm2decoder_options = itertools.product(
    [1, 2], [True, False]
)
lstm2decoder_options = list(lstm2decoder_options)


@pytest.fixture(params=lstm2decoder_options)
def setup_lstm2seqdecoder(request, ):
    HIDDEN_DIM = 1024
    NUM_LAYERS = request.param[0]
    BIDIRECTIONAL = request.param[1]
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    decoder = Lstm2SeqDecoder(
        embedder=embedder,
        vocab_size=10,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        rnn_bias=False,
        num_layers=NUM_LAYERS,
    )

    lines = []
    texts = ["First sentence", "second sentence", "Third long sentence here"]
    for text in texts:
        line = Line(text=text)
        lines.append(line)
    num_direction = 2 if BIDIRECTIONAL else 1
    h0 = torch.ones(NUM_LAYERS, len(texts), num_direction * HIDDEN_DIM) * 0.1
    c0 = torch.ones(NUM_LAYERS, len(texts), num_direction * HIDDEN_DIM) * 0.2

    return (
        decoder,
        {
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_LAYERS": NUM_LAYERS,
            "LINES": lines,
            "TIME_STEPS": 4,
            "VOCAB_SIZE": 10,
            "BIDIRECTIONAL": BIDIRECTIONAL
        },
        (h0, c0)
    )


class TestLstm2SeqDecoder:
    def test_hidden_dim(self, setup_lstm2seqdecoder):
        decoder, options, (h0, c0) = setup_lstm2seqdecoder
        lines = options["LINES"]
        num_time_steps = options["TIME_STEPS"]
        batch_size = len(lines)
        vocab_size = options["VOCAB_SIZE"]
        decoding = decoder(lines=lines, h0=h0, c0=c0)
        assert decoding.size() == (batch_size, num_time_steps, vocab_size)
