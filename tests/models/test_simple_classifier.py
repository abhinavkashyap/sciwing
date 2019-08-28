import pytest

import torch
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
from torch.nn import Embedding
import numpy as np
from sciwing.utils.class_nursery import ClassNursery


@pytest.fixture
def setup_simple_classifier():
    BATCH_SIZE = 1
    NUM_TOKENS = 3
    EMB_DIM = 300
    VOCAB_SIZE = 10
    NUM_CLASSES = 3
    embedding = Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    embedder = VanillaEmbedder(embedding_dim=EMB_DIM, embedding=embedding)
    labels = torch.LongTensor([[1]])
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM, embedder=embedder, dropout_value=0, aggregation_type="sum"
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    simple_classifier = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=False,
    )
    iter_dict = {"tokens": tokens, "label": labels}
    return iter_dict, simple_classifier, BATCH_SIZE, NUM_CLASSES


class TestSimpleClassifier:
    def test_classifier_produces_0_logits_for_0_embedding(
        self, setup_simple_classifier
    ):
        iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
        output = simple_classifier(
            iter_dict, is_training=True, is_validation=False, is_test=False
        )
        logits = output["logits"]
        expected_logits = torch.zeros([batch_size, num_classes])
        assert torch.allclose(logits, expected_logits)

    def test_classifier_produces_equal_probs_for_0_embedding(
        self, setup_simple_classifier
    ):
        iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
        output = simple_classifier(
            iter_dict, is_training=True, is_validation=False, is_test=False
        )
        probs = output["normalized_probs"]
        expected_probs = torch.ones([batch_size, num_classes]) / num_classes
        assert torch.allclose(probs, expected_probs)

    def test_classifier_produces_correct_initial_loss_for_0_embedding(
        self, setup_simple_classifier
    ):
        iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
        output = simple_classifier(
            iter_dict, is_training=True, is_validation=False, is_test=False
        )
        loss = output["loss"].item()
        correct_loss = -np.log(1 / num_classes)
        assert torch.allclose(torch.Tensor([loss]), torch.Tensor([correct_loss]))

    def test_classifier_produces_correct_precision(self, setup_simple_classifier):
        iter_dict, simple_classifier, batch_size, num_classes = setup_simple_classifier
        output = simple_classifier(
            iter_dict, is_training=True, is_validation=False, is_test=False
        )
        idx2labelname_mapping = {0: "good class", 1: "bad class", 2: "average_class"}
        metrics_calc = PrecisionRecallFMeasure(
            idx2labelname_mapping=idx2labelname_mapping
        )

        metrics_calc.calc_metric(iter_dict=iter_dict, model_forward_dict=output)
        metrics = metrics_calc.get_metric()
        precision = metrics["precision"]

        # NOTE: topk returns the last value in the dimension incase
        # all the values are equal.
        expected_precision = {1: 0, 2: 0}

        assert len(precision) == 2

        for class_label, precision_value in precision.items():
            assert precision_value == expected_precision[class_label]

    def test_simple_classifier_in_class_nursery(self):
        assert ClassNursery.class_nursery.get("SimpleClassifier") is not None
