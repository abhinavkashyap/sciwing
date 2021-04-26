import pytest
from sciwing.modules.lstm2seqdecoder import Lstm2SeqDecoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.attentions.dot_product_attention import DotProductAttention
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import (
    AbstractiveSummarizationDatasetManager,
)
from sciwing.data.line import Line
import itertools
import torch
from sciwing.vocab.vocab import Vocab

lstm2decoder_options = itertools.product(
    [1, 2], [True, False], [DotProductAttention(), None], [0, 1]
)
lstm2decoder_options = list(lstm2decoder_options)


@pytest.fixture(params=lstm2decoder_options)
def setup_lstm2seqdecoder(request,):
    HIDDEN_DIM = 1024
    NUM_LAYERS = request.param[0]
    BIDIRECTIONAL = request.param[1]
    TEACHER_FORCING_RATIO = request.param[3]
    MAX_LENGTH = 5

    lines = []
    words = []
    # texts = ["First", "second", "Third"]
    texts = ["First sentence", "second sentence", "Third long sentence here"]
    for text in texts:
        line = Line(text=text)
        word = Line(text=text.split()[0])
        lines.append(line)
        words.append(word)
    flat_texts = [[word for sentence in texts for word in sentence]]
    vocab = Vocab(flat_texts)
    vocab.build_vocab()

    num_direction = 2 if BIDIRECTIONAL else 1
    h0 = torch.ones(NUM_LAYERS, len(texts), num_direction * HIDDEN_DIM) * 0.1
    c0 = torch.ones(NUM_LAYERS, len(texts), num_direction * HIDDEN_DIM) * 0.2

    embedder = WordEmbedder(embedding_type="glove_6B_50")
    encoder_outputs = (
        torch.ones(len(texts), 5, num_direction * HIDDEN_DIM) * 0.5
        if request.param[2]
        else None
    )
    decoder = Lstm2SeqDecoder(
        embedder=embedder,
        vocab=vocab,
        max_length=MAX_LENGTH,
        attn_module=request.param[2],
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        rnn_bias=False,
        num_layers=NUM_LAYERS,
    )

    return (
        decoder,
        {
            "HIDDEN_DIM": HIDDEN_DIM,
            "NUM_LAYERS": NUM_LAYERS,
            "MAX_LENGTH": MAX_LENGTH,
            "TEACHER_FORCING_RATIO": TEACHER_FORCING_RATIO,
            "LINES": lines,
            "WORDS": words,
            "VOCAB_SIZE": vocab.get_vocab_len(),
            "BIDIRECTIONAL": BIDIRECTIONAL,
        },
        encoder_outputs,
        (h0, c0),
    )


class TestLstm2SeqDecoder:
    def test_forward_step(self, setup_lstm2seqdecoder):
        decoder, options, encoder_outputs, (h0, c0) = setup_lstm2seqdecoder
        lines = options["WORDS"]
        num_time_steps = 1
        batch_size = len(lines)
        vocab_size = options["VOCAB_SIZE"]
        decoding, _, _ = decoder.forward_step(
            lines=lines, h0=h0, c0=c0, encoder_outputs=encoder_outputs
        )
        assert decoding.size() == (batch_size, num_time_steps, vocab_size)

    def test_forward(self, setup_lstm2seqdecoder):
        decoder, options, encoder_outputs, (h0, c0) = setup_lstm2seqdecoder
        lines = options["LINES"]
        teacher_forcing_ratio = options["TEACHER_FORCING_RATIO"]
        num_time_steps = (
            max([len(line.tokens["tokens"]) for line in lines])
            if teacher_forcing_ratio > 0
            else options["MAX_LENGTH"]
        )
        batch_size = len(lines)

        vocab_size = options["VOCAB_SIZE"]
        output = decoder.forward(
            lines=lines,
            h0=h0,
            c0=c0,
            encoder_outputs=encoder_outputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        assert output.size() == (batch_size, num_time_steps, vocab_size)
