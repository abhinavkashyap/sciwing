import torch
import torch.nn as nn
from typing import Dict, Any
from torchcrf import CRF
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from torch.nn.functional import softmax


class ParscitTagger(nn.Module):
    def __init__(self, rnn2seqencoder: Lstm2SeqEncoder, hid_dim: int, num_classes: int):
        super(ParscitTagger, self).__init__()

        self.rnn2seqencoder = rnn2seqencoder
        self.hidden_dim = hid_dim
        self.num_classes = num_classes
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
        tokens = iter_dict["tokens"]
        labels = iter_dict["label"]

        assert labels.ndimension() == 2, self.msg_printer(
            f"For Parscit tagger, labels should have 2 dimensions. "
            f"You passed labels that have {labels.ndimension()}"
        )
        batch_size, num_timesteps = tokens.size()

        # batch size, time steps, hidden_dim
        encoding = self.rnn2seqencoder(tokens)

        # batch size, time steps, num_classes
        logits = self.hidden2tag(encoding)

        normalized_probs = softmax(logits, dim=2)

        output_dict = {
            "logits": logits.view(batch_size * num_timesteps, -1),
            "normalized_probs": normalized_probs.view(batch_size * num_timesteps, -1),
        }

        if is_training or is_validation:
            loss = self.crf(logits, labels)
            output_dict["loss"] = loss

        return output_dict
