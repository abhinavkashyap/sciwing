import torch.nn as nn
from parsect.modules.bow_elmo_encoder import BowElmoEncoder
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
import wasabi
from typing import Dict, Any


class BowElmoLinearClassifier(nn.Module):
    def __init__(self,
                 encoder: BowElmoEncoder,
                 encoding_dim: int,
                 num_classes: int,
                 classification_layer_bias: bool = True):
        super(BowElmoLinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.classification_layer_bias = classification_layer_bias

        self.classification_layer = nn.Linear(
            encoding_dim, num_classes, bias=self.classification_layer_bias
        )
        self._loss = CrossEntropyLoss()
        self.msg_printer = wasabi.Printer()

    def forward(self,
                iter_dict: Dict[str, Any],
                is_training: bool,
                is_validation: bool,
                is_test: bool) -> Dict[str, Any]:
        labels = iter_dict['label']
        labels = labels.squeeze(1)
        x = iter_dict['instance']
        x = [instance.split() for instance in x]

        assert labels.ndimension() == 1, self.msg_printer.fail(
            "the labels should have 1 dimension "
            "your input has shape {0}".format(labels.size())
        )

        encoding = self.encoder(x)

        # N * C
        # N - batch size
        # C - number of classes
        logits = self.classification_layer(encoding)

        # N * C
        # N - batch size
        # C - number of classes
        # The normalized probabilities of classification
        normalized_probs = softmax(logits, dim=1)

        output_dict = {"logits": logits, "normalized_probs": normalized_probs}

        if is_training or is_validation:
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss

        return output_dict
