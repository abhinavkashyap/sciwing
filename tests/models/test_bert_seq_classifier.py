import pytest
import torch
from parsect.models.bert_seq_classifier import BertSeqClassifier
from parsect.tokenizers.bert_tokenizer import TokenizerForBert
from parsect.utils.common import pack_to_length


@pytest.fixture
def setup_bert_seq_classifier():
    num_classes = 10
    bert_type = "bert-base-uncased"
    emb_dim = 768
    dropout_value = 0.0
    batch_size = 3

    classifier = BertSeqClassifier(
        num_classes=num_classes,
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        bert_type=bert_type,
    )

    tokenizer = TokenizerForBert(bert_type=bert_type)

    raw_instance = [
        "i like the bert model",
        "sci bert is better",
        "hugging face makes it easier",
    ]
    tokenized_text = []
    for instance in raw_instance:
        tokenized_text.append(tokenizer.tokenize(instance))

    padded_instances = []
    for instance in tokenized_text:
        padded_instance = pack_to_length(
            tokenized_text=instance,
            max_length=10,
            pad_token="[PAD]",
            add_start_end_token=True,
            start_token="[CLS]",
            end_token="[SEP]",
        )
        padded_instances.append(padded_instance)

    bert_tokens = list(map(tokenizer.convert_tokens_to_ids, padded_instances))
    bert_tokens = torch.LongTensor(bert_tokens)
    segment_ids = [[0] * len(instance) for instance in padded_instances]
    segment_ids = torch.LongTensor(segment_ids)
    labels = torch.LongTensor([[1], [2], [3]])
    iter_dict = {
        "raw_instance": raw_instance,
        "label": labels,
        "bert_tokens": bert_tokens,
        "segment_ids": segment_ids,
    }
    options = {"num_classes": num_classes, "batch_size": batch_size}

    return classifier, iter_dict, options


@pytest.fixture
def setup_scibert_seq_classifier():
    num_classes = 10
    bert_type = "scibert-base-uncased"
    emb_dim = 768
    dropout_value = 0.0
    batch_size = 3

    classifier = BertSeqClassifier(
        num_classes=num_classes,
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        bert_type=bert_type,
    )

    tokenizer = TokenizerForBert(bert_type=bert_type)

    raw_instance = [
        "i like the bert model",
        "sci bert is better",
        "hugging face makes it easier",
    ]
    tokenized_text = []
    for instance in raw_instance:
        tokenized_text.append(tokenizer.tokenize(instance))

    padded_instances = []
    for instance in tokenized_text:
        padded_instance = pack_to_length(
            tokenized_text=instance,
            max_length=10,
            pad_token="[PAD]",
            add_start_end_token=True,
            start_token="[CLS]",
            end_token="[SEP]",
        )
        padded_instances.append(padded_instance)

    bert_tokens = list(map(tokenizer.convert_tokens_to_ids, padded_instances))
    bert_tokens = torch.LongTensor(bert_tokens)
    segment_ids = [[0] * len(instance) for instance in padded_instances]
    segment_ids = torch.LongTensor(segment_ids)
    labels = torch.LongTensor([[1], [2], [3]])
    iter_dict = {
        "raw_instance": raw_instance,
        "label": labels,
        "bert_tokens": bert_tokens,
        "segment_ids": segment_ids,
    }
    options = {"num_classes": num_classes, "batch_size": batch_size}

    return classifier, iter_dict, options


class TestBertSeqClassifier:
    def test_bert_seq_classifier_dimensions(self, setup_bert_seq_classifier):
        classifier, iter_dict, options = setup_bert_seq_classifier
        output_dict = classifier(
            iter_dict=iter_dict, is_training=True, is_validation=False, is_test=False
        )

        batch_size = options["batch_size"]
        num_classes = options["num_classes"]

        assert output_dict["normalized_probs"].size() == (batch_size, num_classes)

    def test_scibert_seq_classifier_dimensions(self, setup_scibert_seq_classifier):
        classifier, iter_dict, options = setup_scibert_seq_classifier
        output_dict = classifier(
            iter_dict=iter_dict, is_training=True, is_validation=False, is_test=False
        )

        batch_size = options["batch_size"]
        num_classes = options["num_classes"]

        assert output_dict["normalized_probs"].size() == (batch_size, num_classes)
