import pytest
from sciwing.models.parscit_tagger import ParscitTagger
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
from sciwing.modules.charlstm_encoder import CharLSTMEncoder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
import itertools
import torch.nn as nn
import torch
import numpy as np

lstm2encoder_options = itertools.product(
    [True, False], ["sum", "concat"], [True, False]
)
lstm2encoder_options = list(lstm2encoder_options)


@pytest.fixture(params=lstm2encoder_options)
def setup_parscit_tagger(request):
    EMBEDDING_DIM = 100
    CHARACTER_EMBEDDING_DIM = 25
    VOCAB_SIZE = 1000
    BATCH_SIZE = 2
    HIDDEN_DIM = 1024
    CHARACTER_ENCODER_HIDDEN_DIM = 100
    NUM_TIME_STEPS = 10
    MAX_CHAR_LENGTH = 25
    CHAR_VOCAB_SIZE = 100
    BIDIRECTIONAL = request.param[0]
    COMBINE_STRATEGY = request.param[1]
    HAVE_CHARACTER_ENCODER = request.param[2]
    NUM_CLASSES = 5
    EMBEDDING = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMBEDDING_DIM]))
    CHARACTER_EMBEDDING = nn.Embedding.from_pretrained(
        torch.zeros([CHAR_VOCAB_SIZE, CHARACTER_EMBEDDING_DIM])
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TIME_STEPS))
    labels = np.random.randint(0, NUM_CLASSES - 1, size=(BATCH_SIZE, NUM_TIME_STEPS))
    char_tokens = np.random.randint(
        0, CHAR_VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TIME_STEPS, MAX_CHAR_LENGTH)
    )
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor(labels)
    char_tokens = torch.LongTensor(char_tokens)

    embedder = VanillaEmbedder(embedding=EMBEDDING, embedding_dim=EMBEDDING_DIM)
    if HAVE_CHARACTER_ENCODER:
        char_embedder = VanillaEmbedder(
            embedding=CHARACTER_EMBEDDING, embedding_dim=CHARACTER_EMBEDDING_DIM
        )
        char_encoder = CharLSTMEncoder(
            char_embedder=char_embedder,
            char_emb_dim=CHARACTER_EMBEDDING_DIM,
            hidden_dim=CHARACTER_ENCODER_HIDDEN_DIM,
            bidirectional=True,
            combine_strategy="concat",
        )
        embedder = ConcatEmbedders([embedder, char_encoder])
        EMBEDDING_DIM = EMBEDDING_DIM + (2 * CHARACTER_ENCODER_HIDDEN_DIM)

    encoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIM,
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
    )

    tagger = ParscitTagger(
        rnn2seqencoder=encoder,
        hid_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )

    return (
        tagger,
        {
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "VOCAB_SIZE": VOCAB_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "HIDDEN_DIM": 2 * HIDDEN_DIM
            if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
            else HIDDEN_DIM,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "tokens": tokens,
            "labels": labels,
            "EXPECTED_HIDDEN_DIM": 2 * HIDDEN_DIM
            if COMBINE_STRATEGY == "concat" and BIDIRECTIONAL
            else HIDDEN_DIM,
            "TIME_STEPS": NUM_TIME_STEPS,
            "NUM_CLASSES": NUM_CLASSES,
            "HAVE_CHAR_ENCODER": HAVE_CHARACTER_ENCODER,
            "char_tokens": char_tokens,
        },
    )


class TestParscitTagger:
    def test_parscit_tagger_dimensions(self, setup_parscit_tagger):
        tagger, options = setup_parscit_tagger
        tokens = options["tokens"]
        labels = options["labels"]
        char_tokens = options["char_tokens"]
        BATCH_SIZE = options["BATCH_SIZE"]
        TIME_STEPS = options["TIME_STEPS"]
        NUM_CLASSES = options["NUM_CLASSES"]

        iter_dict = {"tokens": tokens, "label": labels, "char_tokens": char_tokens}

        output_dict = tagger(
            iter_dict=iter_dict, is_training=True, is_validation=False, is_test=False
        )

        assert output_dict["logits"].size() == (BATCH_SIZE, TIME_STEPS, NUM_CLASSES)
