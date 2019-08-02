import torch.nn as nn
from typing import Dict, Any, Union
from torchcrf import CRF
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder


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

        # batch size, time steps, hidden_dim
        encoding = self.rnn2seqencoder(iter_dict=iter_dict)

        # batch size, time steps, num_classes
        logits = self.hidden2tag(encoding)

        predicted_tags = self.crf.decode(logits)

        output_dict = {"logits": logits, "predicted_tags": predicted_tags}

        if is_training or is_validation:
            labels = iter_dict["label"]
            assert labels.ndimension() == 2, self.msg_printer(
                f"For Parscit tagger, labels should have 2 dimensions. "
                f"You passed labels that have {labels.ndimension()}"
            )
            loss = -self.crf(logits, labels)
            output_dict["loss"] = loss

        return output_dict
