from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from typing import Dict, Any
import torch.nn as nn
import wasabi
import torch
import os
import parsect.constants as constants

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]


class BertSeqClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 768,
        dropout_value: float = 0.0,
        bert_type: str = "bert-base-uncased",
        device: torch.device = torch.device("cpu"),
    ):
        """
        """
        super(BertSeqClassifier, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.bert_type = bert_type
        self.device = device
        self.msg_printer = wasabi.Printer()

        self.allowed_bert_types = [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "scibert-base-cased",
            "scibert-sci-cased",
            "scibert-base-uncased",
            "scibert-sci-uncased",
        ]
        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }
        self.model_type_or_folder_url = None
        self.vocab_type_or_filename = None

        assert self.bert_type in self.allowed_bert_types

        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = os.path.join(
                MODELS_CACHE_DIR, foldername, "weights.tar.gz"
            )
            self.vocab_type_or_filename = os.path.join(
                MODELS_CACHE_DIR, foldername, "vocab.txt"
            )
        else:
            self.model_type_or_folder_url = self.bert_type
            self.vocab_type_or_filename = self.bert_type

        # load the bert model
        with self.msg_printer.loading("Loading Bert model"):
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_type_or_folder_url, self.num_classes
            )

        self.msg_printer.good(f"Finished Loading {self.bert_type} tokenizer and model")

        self._loss = CrossEntropyLoss()

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
        tokens_tensor = iter_dict["bert_tokens"]
        segment_tensor = iter_dict["segment_ids"]
        labels = iter_dict["label"]
        labels = labels.squeeze(1)

        logits = self.model(input_ids=tokens_tensor, token_type_ids=segment_tensor)

        normalized_probs = softmax(logits, dim=1)

        output_dict = {"logits": logits, "normalized_probs": normalized_probs}

        if is_training or is_validation:
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss

        return output_dict
