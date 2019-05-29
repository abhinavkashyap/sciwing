import pytest
from parsect.modules.elmo_embedder import ElmoEmbedder
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
from parsect.models.elmo_lstm_classifier import ElmoLSTMClassifier
import torch.nn as nn
import torch
import numpy as np


@pytest.fixture
def setup_elmo_classifier():
    elmo_embedder = ElmoEmbedder()
    elmo_emb_dim = 1024
    emb_dim = 100
    vocab_size = 100
    embedding = nn.Embedding(vocab_size, emb_dim)
    dropout_value = 0.0
    hidden_dim = 512
    bidirectional = True
    combine_strategy = "concat"
    BATCH_SIZE = 1
    NUM_TOKENS = 5
    NUM_CLASSES = 10
    tokens = np.random.randint(0, vocab_size - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor([[1]])
    instances = ["i like to test this".split()]

    elmo_lstm_encoder = ElmoLSTMEncoder(
        elmo_emb_dim=elmo_emb_dim,
        elmo_embedder=elmo_embedder,
        emb_dim=emb_dim,
        embedding=embedding,
        dropout_value=dropout_value,
        hidden_dim=hidden_dim,
        bidirectional=bidirectional,
        combine_strategy=combine_strategy,
        rnn_bias=True,
    )

    elmo_lstm_classifier = ElmoLSTMClassifier(
        elmo_lstm_encoder=elmo_lstm_encoder,
        encoding_dim=2 * hidden_dim,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    options = {
        "ELMO_EMB_DIM": elmo_emb_dim,
        "EMB_DIM": emb_dim,
        "VOCAB_SIZE": vocab_size,
        "HIDDEN_DIM": hidden_dim,
        "BIDIRECTIONAL": bidirectional,
        "COMBINE_STRATEGY": combine_strategy,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_CLASSES": NUM_CLASSES
    }

    iter_dict = {"tokens": tokens, "label": labels, "instance": instances}

    return elmo_lstm_classifier, iter_dict, options


class TestElmoLSTMClassifier:
    def test_elmo_lstm_classifier_dimension(self, setup_elmo_classifier):
        elmo_lstm_classifier, iter_dict, options = setup_elmo_classifier
        output_dict = elmo_lstm_classifier(
            iter_dict=iter_dict, is_training=True, is_validation=True, is_test=True
        )
        batch_size = options['BATCH_SIZE']
        num_classes = options["NUM_CLASSES"]

        assert output_dict['logits'].size() == (batch_size, num_classes)
