from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from typing import Dict, Any
import torch.nn as nn
import wasabi
import torch
from parsect.utils.common import pack_to_length


class BertSeqClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 768,
        dropout_value: float = 0.0,
        bert_type: str = "bert-base-uncased",
    ):
        """
        """
        super(BertSeqClassifier, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.bert_type = bert_type
        self.msg_printer = wasabi.Printer()

        self.allowed_bert_types = [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
        ]

        assert self.bert_type in self.allowed_bert_types

        # load the bert model
        with self.msg_printer.loading("Loading Bert tokenizer and model"):
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_type)
            self.model = BertForSequenceClassification.from_pretrained(
                self.bert_type, self.num_classes
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
        raw_instance = iter_dict["raw_instance"]
        labels = iter_dict["label"]
        labels = labels.squeeze(1)

        batch_size = len(raw_instance)

        tokenized_text = list(map(self.bert_tokenizer.tokenize, raw_instance))
        lengths = list(map(lambda tokenized: len(tokenized), tokenized_text))
        max_len = sorted(lengths, reverse=True)[0]

        # avoid attention over the padded tokens
        # 1 is to signify "attend" and "0" signifies dont attend
        attention_masks = [
            [1] * length + [0] * (max_len - length) for length in lengths
        ]

        # pad the tokenized text to a maximum length
        padded_tokenized_text = []
        for tokens in tokenized_text:
            padded_tokens = pack_to_length(
                tokenized_text=tokens,
                max_length=max_len,
                pad_token="[PAD]",
                add_start_end_token=True,
                start_token="[CLS]",
                end_token="[SEP]",
            )
            padded_tokenized_text.append(padded_tokens)

        # convert them to ids based on bert vocab
        indexed_tokens = list(
            map(self.bert_tokenizer.convert_tokens_to_ids, padded_tokenized_text)
        )
        segment_ids = list(
            map(lambda tokens_list: [0] * len(tokens_list), indexed_tokens)
        )

        tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long)
        segment_tensor = torch.tensor(segment_ids, dtype=torch.long)
        attention_masks = torch.LongTensor(attention_masks)

        assert tokens_tensor.size() == (batch_size, max_len)
        assert segment_tensor.size() == (batch_size, max_len)
        assert attention_masks.size() == (batch_size, max_len)

        logits = self.model(
            input_ids=tokens_tensor,
            token_type_ids=segment_tensor,
            attention_mask=attention_masks,
        )

        normalized_probs = softmax(logits, dim=1)

        output_dict = {"logits": logits, "normalized_probs": normalized_probs}

        if is_training or is_validation:
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss

        return output_dict
