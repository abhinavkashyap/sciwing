import torch.nn as nn
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
import wasabi
from typing import Dict, Any
import torch


class BowElmoLinearClassifier(nn.Module):
    """Bag of words Elmo representations followed by Linear Classifier
    """

    def __init__(
        self,
        encoder: BowElmoEmbedder,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool = True,
    ):
        """
        Parameters
        ----------
        encoder : BowElmoEmbedder
            Bag of words Elmo Embedder, that embeds words by aggregating
            elmo representations across words either by summing or averaging
            or any other strategy
        encoding_dim : int
            Dimension of the encoding of text
        num_classes : int
            Number of classes in the dataset
        classification_layer_bias : bool
            whether to add bias to the classification layer
            This is set to false only for testing or debugging purpose
            Else please keep this as true
        """
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
            Expects ``instance`` to be present in the ``iter_dict``
            where the instance is a space separated text that
            is used by the ElmoEmbedder

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

        x = iter_dict["instance"]
        x = [instance.split() for instance in x]

        encoding = self.encoder(x)

        # TODO: quick fix for cuda situation
        # ideally have to test by converting the instance to cuda

        encoding = encoding.cuda() if torch.cuda.is_available() else encoding

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
