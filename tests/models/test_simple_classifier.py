import pytest
import torch
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simple_classifier import Simple_Classifier
from torch.nn import Embedding
import numpy as np


@pytest.fixture
def setup_classifier_bs_1():
    BATCH_SIZE = 1
    NUM_TOKENS = 3
    EMB_DIM = 300
    VOCAB_SIZE = 10
    NUM_CLASSES = 3
    embedding = Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    labels = torch.LongTensor([1])
    encoder = BOW_Encoder(emb_dim=EMB_DIM,
                          embedding=embedding,
                          dropout_value=0,
                          aggregation_type='sum')
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    simple_classifier = Simple_Classifier(encoder=encoder,
                                          encoding_dim=EMB_DIM,
                                          num_classes=NUM_CLASSES,
                                          classification_layer_bias=False
                                          )
    return tokens, labels, simple_classifier, BATCH_SIZE, NUM_CLASSES


class TestSimpleClassifier:
    def test_classifier_produces_0_logits_for_0_embedding(self, setup_classifier_bs_1):
        tokens, labels, simple_classifier, batch_size, num_classes = setup_classifier_bs_1
        output = simple_classifier(tokens, labels, is_training=True)
        logits = output['logits']
        expected_logits = torch.zeros([batch_size, num_classes])
        assert torch.allclose(logits, expected_logits)

    def test_classifier_produces_equal_probs_for_0_embedding(self, setup_classifier_bs_1):
        tokens, labels, simple_classifier, batch_size, num_classes = setup_classifier_bs_1
        output = simple_classifier(tokens, labels, is_training=True)
        probs = output['normalized_probs']
        expected_probs = torch.ones([batch_size, num_classes]) / num_classes
        assert torch.allclose(probs, expected_probs)

    def test_classifier_produces_correct_initial_loss_for_0_embedding(self, setup_classifier_bs_1):
        tokens, labels, simple_classifier, batch_size, num_classes = setup_classifier_bs_1
        output = simple_classifier(tokens, labels, is_training=True)
        loss = output['loss'].item()
        correct_loss = -np.log(1/num_classes)
        assert torch.allclose(torch.Tensor([loss]), torch.Tensor([correct_loss]))

    def test_classifier_produces_correct_precision(self, setup_classifier_bs_1):
        tokens, labels, simple_classifier, batch_size, num_classes = setup_classifier_bs_1
        output = simple_classifier(tokens, labels, is_training=True)
        precision = output['precision']
        expected_precision = {1: 0, 2: 0}

        assert len(precision) == 2

        for class_label, precision_value in precision.items():
            print(class_label)
            assert precision_value == expected_precision[class_label]
