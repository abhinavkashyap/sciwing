import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import Dict, Any
from wasabi import Printer
from sciwing.utils.class_nursery import ClassNursery


class SimpleClassifier(nn.Module, ClassNursery):
    def __init__(
        self,
        encoder: nn.Module,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool,
    ):
        super(SimpleClassifier, self).__init__()
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        print(self.num_classes)
        self.classification_layer_bias = classification_layer_bias
        self.classification_layer = nn.Linear(
            encoding_dim, num_classes, bias=self.classification_layer_bias
        )
        self._loss = CrossEntropyLoss()
        self.msg_printer = Printer()

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ) -> Dict[str, Any]:
        encoding = self.encoder(iter_dict=iter_dict)

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
            labels = iter_dict["label"]
            labels = labels.squeeze(1)
            assert labels.ndimension() == 1, self.msg_printer.fail(
                "the labels should have 1 dimension "
                "your input has shape {0}".format(labels.size())
            )
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss

        return output_dict
