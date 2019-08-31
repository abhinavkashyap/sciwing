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
        """ SimpleClassifier is a linear classifier head on top of any encoder

        Parameters
        ----------
        encoder : nn.Module
            Any encoder that takes in instances
        encoding_dim : int
            The encoding dimension
        num_classes : int
            The number of classes
        classification_layer_bias : bool
            Whether to add classification layer bias or no
            This is set to false only for debugging purposes ff
        """
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
        """

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` from any dataset that will be passed on to the encoder
        is_training : bool
            running forward on training dataset?
        is_validation : bool
            running forward on training dataset ?
        is_test : bool
            running forward on test dataset?


        Returns
        -------
        Dict[str, Any]
            logits: torch.FloatTensor
                Un-normalized probabilities over all the classes
                of the shape ``[batch_size, num_classes]``
            normalized_probs: torch.FloatTensor
                Normalized probabilities over all the classes
                of the shape ``[batch_size, num_classes]``
            loss: float
                Loss value if this is a training forward pass
                or validation loss. There will be no loss
                if this is the test dataset

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
