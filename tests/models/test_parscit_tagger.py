import pytest
from parsect.models.parscit_tagger import ParscitTagger
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
import itertools
import torch.nn as nn
import torch
import numpy as np

lstm2encoder_options = itertools.product([True, False], ["sum", "concat"])
lstm2encoder_options = list(lstm2encoder_options)


@pytest.fixture(params=lstm2encoder_options)
def setup_parscit_tagger(request):
    EMBEDDING_DIM = 100
    VOCAB_SIZE = 1000
    BATCH_SIZE = 2
    HIDDEN_DIM = 1024
    NUM_TIME_STEPS = 10
    BIDIRECTIONAL = request.param[0]
    COMBINE_STRATEGY = request.param[1]
    NUM_CLASSES = 5
    EMBEDDING = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMBEDDING_DIM]))
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TIME_STEPS))
    labels = np.random.randint(0, NUM_CLASSES - 1, size=(BATCH_SIZE, NUM_TIME_STEPS))
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor(labels)

    encoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIM,
        embedding=EMBEDDING,
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
        },
    )


class TestParscitTagger:
    def test_parscit_tagger_dimensions(self, setup_parscit_tagger):
        tagger, options = setup_parscit_tagger
        tokens = options["tokens"]
        labels = options["labels"]
        BATCH_SIZE = options["BATCH_SIZE"]
        TIME_STEPS = options["TIME_STEPS"]
        NUM_CLASSES = options["NUM_CLASSES"]

        iter_dict = {"tokens": tokens, "label": labels}

        output_dict = tagger(
            iter_dict=iter_dict, is_training=True, is_validation=False, is_test=False
        )

        assert output_dict["logits"].size() == (BATCH_SIZE, TIME_STEPS, NUM_CLASSES)
