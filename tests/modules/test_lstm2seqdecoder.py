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

@pytest.fixture(scope="session")
def abs_sum_dataset_manager(tmpdir_factory, request):
    train_file = tmpdir_factory.mktemp("train_data").join("train_file.txt")
    train_file.write("train_word1 train_word2###label1\ntrain_word3###label2")

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev_file.txt")
    dev_file.write("dev_word1###label1\ndev_word2###label2")

    test_file = tmpdir_factory.mktemp("test_data").join("test_file.txt")
    test_file.write("test_word1###label1\ntest_word2###label2")

    abs_sum_dataset_manager = AbstractiveSummarizationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return abs_sum_dataset_manager


@pytest.fixture(params=lstm2decoder_options)
def setup_lstm2seqdecoder(request, abs_sum_dataset_manager):
    HIDDEN_DIM = 1024
    NUM_LAYERS = request.param[0]
    OUTPUT_DIM = abs_sum_dataset_manager.namespace_to_vocab["tokens"].get_vocab_len()
    BIDIRECTIONAL = request.param[1]
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    decoder = Lstm2SeqDecoder(
        embedder=embedder,
        datasets_manager=abs_sum_dataset_manager,
        word_tokens_namespace="tokens",
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
            "EXPECTED_OUTPUT_DIM": OUTPUT_DIM,
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
        expected_output_size = options["EXPECTED_OUTPUT_DIM"]
        bidirectional = options["BIDIRECTIONAL"]
        decoding = decoder(lines=lines)
        batch_size = len(lines)
        hidden_dim = 2 * options["HIDDEN_DIM"] if bidirectional else options["HIDDEN_DIM"]
        assert decoding.size() == (batch_size, num_time_steps, expected_output_size)
        assert decoder.hidden_dim == hidden_dim
