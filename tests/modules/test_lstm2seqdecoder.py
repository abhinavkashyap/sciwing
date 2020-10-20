import pytest
from sciwing.modules.lstm2seqdecoder import Lstm2SeqDecoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import AbstractiveSummarizationDatasetManager
from sciwing.data.line import Line
import itertools

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
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        rnn_bias=False,
        num_layers=NUM_LAYERS,
    )

    lines = []
    texts = ["First sentence", "second sentence"]
    for text in texts:
        line = Line(text=text)
        lines.append(line)

    return (
        decoder,
        {
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_LAYERS": NUM_LAYERS,
            "LINES": lines,
            "TIME_STEPS": 2,
            "BIDIRECTIONAL": BIDIRECTIONAL
        },
    )


class TestLstm2SeqDecoder:
    def test_hidden_dim(self, setup_lstm2seqdecoder):
        decoder, options = setup_lstm2seqdecoder
        lines = options["LINES"]
        num_time_steps = options["TIME_STEPS"]
        bidirectional = options["BIDIRECTIONAL"]
        expected_output_size =  2 * options["HIDDEN_DIM"] if bidirectional else options["HIDDEN_DIM"]
        decoding = decoder(lines=lines)
        batch_size = len(lines)
        assert decoding.size() == (batch_size, num_time_steps, expected_output_size)
