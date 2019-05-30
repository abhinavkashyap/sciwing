import pytest
from parsect.modules.bow_bert_encoder import BowBertEncoder
from parsect.models.bow_bert_linear_classifier import BowBertLinearClassifier
import torch


@pytest.fixture()
def setup_test_bow_bert_lc():
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "sum"
    bert_type = "bert-base-uncased"
    num_classes = 10
    batch_size = 1

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )

    classifier = BowBertLinearClassifier(
        encoder=bow_bert_encoder, encoding_dim=emb_dim, num_classes=num_classes
    )

    raw_instance = ["this is a bow test"]
    label = torch.LongTensor([[1]])

    iter_dict = {"raw_instance": raw_instance, "label": label}

    options = {
        "EMB_DIM": emb_dim,
        "DROPOUT": dropout_value,
        "AGGREGATION_TYPE": aggregation_type,
        "BERT_TYPE": bert_type,
        "NUM_CLASSES": num_classes,
        "BATCH_SIZE": batch_size,
    }

    return classifier, iter_dict, options


class TestBowBertLinearClassifier:
    def test_bow_bert_linear_classifier_dimension(self, setup_test_bow_bert_lc):
        classifier, iter_dict, options = setup_test_bow_bert_lc

        output_dict = classifier(
            iter_dict=iter_dict, is_training=True, is_validation=True, is_test=True
        )

        assert output_dict["logits"].size() == (
            options["BATCH_SIZE"],
            options["NUM_CLASSES"],
        )
