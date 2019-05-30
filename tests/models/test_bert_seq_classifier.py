import pytest
import torch
from parsect.models.bert_seq_classifier import BertSeqClassifier


@pytest.fixture
def setup_bert_seq_classifier():
    num_classes = 10
    bert_type = "bert-base-uncased"
    emb_dim = 768
    dropout_value = 0.0
    batch_size = 3

    raw_instance = [
        "i like the bert model",
        "sci bert is better",
        "hugging face makes it easier",
    ]
    labels = torch.LongTensor([[1], [2], [3]])
    iter_dict = {"raw_instance": raw_instance, "label": labels}
    options = {"num_classes": num_classes, "batch_size": batch_size}
    classifier = BertSeqClassifier(
        num_classes=num_classes,
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        bert_type=bert_type,
    )

    return classifier, iter_dict, options


class TestBertSeqClassifier:
    def test_bert_seq_classifier_dimensions(self, setup_bert_seq_classifier):
        classifier, iter_dict, options = setup_bert_seq_classifier
        output_dict = classifier(
            iter_dict=iter_dict, is_training=True, is_validation=True, is_test=True
        )

        batch_size = options["batch_size"]
        num_classes = options["num_classes"]

        assert output_dict["normalized_probs"].size() == (batch_size, num_classes)
