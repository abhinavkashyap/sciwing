import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import Dict, Any
from wasabi import Printer


class SimpleClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool,
    ):
        """

        :param encoder: type: nn.Module
        Any encoder that encodes the input to a single vector
        :param encoding_dim type: int
        The encoding dimension i
        :param num_classes: type: int
        The number of classes in the output
        :param classification_layer_bias: type: bool
        Would you want to add bias to the classification layer
        This can be useful for testing the classifier
        """
        super(SimpleClassifier, self).__init__()
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
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
        """
        :param iter_dict: type: Dict[str, Any]
        iter dict is the dict that is returned by a dataset
        :param is_training: type: bool
        indicates whether the forward method is being called for training
        inn which case we have to calculate a loss or during inference time
        when just the probabilities are returned
        :param is_test: type: bool
        Is this model being run in the inference mode
        :param is_validation: type: bool
        Is this model beign run in the validation model
        :return: type: Dict[str, Any]
        """

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
