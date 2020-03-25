import pytest
from sciwing.modules.lstm2seq_attncontext_encoder import Lstm2SeqAttnContextEncoder
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.attentions.dot_product_attention import DotProductAttention
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.data.contextual_lines import LineWithContext


@pytest.fixture
def encoder():
    embedder = WordEmbedder(embedding_type="glove_6B_50")
    hidden_dim = 50

    lstm2seqencoder = Lstm2SeqEncoder(
        embedder=embedder, hidden_dim=hidden_dim, bidirectional=False
    )

    attn_module = DotProductAttention()

    context_embedder = WordEmbedder(embedding_type="glove_6B_50")

    encoder = Lstm2SeqAttnContextEncoder(
        rnn2seqencoder=lstm2seqencoder,
        attn_module=attn_module,
        context_embedder=context_embedder,
    )

    return encoder


@pytest.fixture
def data():
    text = "This is a string"
    context = ["Context string first", "Context string second lah!"]
    line = LineWithContext(text=text, context=context)
    return [line]


class TestLSTM2SeqAttnContextEncoder:
    def test_encoding_size(self, encoder, data):
        encoding = encoder(data)
        assert encoding.size(0) == 1
        assert encoding.size(1) == 4
        assert encoding.size(2) == 100
