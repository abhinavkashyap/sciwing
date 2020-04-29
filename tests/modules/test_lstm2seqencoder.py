import pytest
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.line import Line
import itertools

add_projection_layer = [True, False]
lstm2encoder_options = itertools.product(
    [True, False], ["sum", "concat"], [1, 2], add_projection_layer
)
lstm2encoder_options = list(lstm2encoder_options)


@pytest.fixture(params=lstm2encoder_options)
def setup_lstm2seqencoder(request):
    HIDDEN_DIM = 1024
    BIDIRECTIONAL = request.param[0]
    COMBINE_STRATEGY = request.param[1]
    NUM_LAYERS = request.param[2]
    ADD_PROJECTION_LAYER = request.param[3]
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    encoder = Lstm2SeqEncoder(
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
        num_layers=NUM_LAYERS,
        add_projection_layer=ADD_PROJECTION_LAYER,
    )

    lines = []
    texts = ["First sentence", "second sentence"]
    for text in texts:
        line = Line(text=text)
        lines.append(line)

    return (
        encoder,
        {
            "HIDDEN_DIM": HIDDEN_DIM,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "EXPECTED_HIDDEN_DIM": 2 * HIDDEN_DIM
            if COMBINE_STRATEGY == "concat"
            and BIDIRECTIONAL
            and not ADD_PROJECTION_LAYER
            else HIDDEN_DIM,
            "NUM_LAYERS": NUM_LAYERS,
            "LINES": lines,
            "TIME_STEPS": 2,
        },
    )


class TestLstm2SeqEncoder:
    def test_hidden_dim(self, setup_lstm2seqencoder):
        encoder, options = setup_lstm2seqencoder
        lines = options["LINES"]
        num_time_steps = options["TIME_STEPS"]
        expected_hidden_size = options["EXPECTED_HIDDEN_DIM"]
        encoding = encoder(lines=lines)
        batch_size = len(lines)
        assert encoding.size() == (batch_size, num_time_steps, expected_hidden_size)
