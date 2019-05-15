import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import Dict, Any
from wasabi import Printer
from parsect.metrics.accuracy_metrics import Accuracy


class Simple_Classifier(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 encoding_dim: int,
                 num_classes: int,
                 classification_layer_bias: bool):
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
        super(Simple_Classifier, self).__init__()
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.classification_layer_bias = classification_layer_bias
        self.classification_layer = nn.Linear(encoding_dim, num_classes,
                                              bias=self.classification_layer_bias)
        self._loss = CrossEntropyLoss()
        self.accuracy_calculator = Accuracy()
        self.msg_printer = Printer()

    def forward(self, x: torch.LongTensor,
                labels: torch.LongTensor,
                is_training: bool) -> Dict[str, Any]:
        """
        :param x: type: torch.LongTensor
                  shape: N * T
                  N - batch size
                  T - Number of tokens per batch
        :param labels: type: torch.LongTensor
                shape: N,
                N - batch size
        :param is_training: type: bool
        indicates whether the forward method is being called for training
        inn which case we have to calculate a loss or during inference time
        when just the probabilities are returne
        :return: type: Dict[str, Any]
        """

        # N * D
        # N - batch size
        # D - Encoding dimension `self.encoding_dim`

        assert x.ndimension() == 2, self.msg_printer.fail('the input should have 2 dimensions  d'
                                                          'your input has shape {0}'
                                                          .format(x.size()))
        assert labels.ndimension() == 1, self.msg_printer.fail('the labels should have 1 dimension '
                                                               'your input has shape {0}'
                                                               .format(labels.size()))

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

        output_dict = {
            'logits': logits,
            'normalized_probs': normalized_probs
        }

        if is_training:
            print('labels size', labels.size())
            loss = self._loss(logits, labels)
            output_dict['loss'] = loss

        # calculate metrics
        metrics = self.accuracy_calculator.get_accuracy(
            normalized_probs, labels
        )

        # combine the two dicts
        output_dict = {**output_dict, **metrics}

        return output_dict


