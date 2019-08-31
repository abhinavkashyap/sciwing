import torch.nn as nn
from sciwing.modules.elmo_lstm_encoder import ElmoLSTMEncoder
import wasabi
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import Dict, Any
import torch
from deprecated import deprecated


@deprecated(
    reason="ELMO LSTM Classifier can be composed using embedders. This "
    "class will be removed in version 0.2"
)
class ElmoLSTMClassifier(nn.Module):
    """Classifier head for `ElmoLSTMEncoder` which concatenates
    the elmo embeddings and token embeddings and passes it through
    another lstm layer
    """

    def __init__(
        self,
        elmo_lstm_encoder: ElmoLSTMEncoder,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        .. deprecated:: 0.0.1
            This can be composed easily using embedders.

        Parameters
        ----------
        elmo_lstm_encoder : ElmoLSTMEncoder
            ElmoLSTMEncoder that combines the token embedding
            and elmo embedding and then passes it through another LSTM
        encoding_dim : int
            The encoding dimension from the ElmoLSTMEncoder
        num_classes : int
            Number of classes of the dataset
        classification_layer_bias : bool
            Whether to turn on the classification layer bias or no
        device : torch.device
            The device on which the model is placed

        """
        super(ElmoLSTMClassifier, self).__init__()
        self.elmo_lstm_encoder: ElmoLSTMEncoder = elmo_lstm_encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.classification_layer_bias = classification_layer_bias
        self.classification_layer = nn.Linear(
            encoding_dim, num_classes, bias=self.classification_layer_bias
        )
        self.device = device
        self._loss = CrossEntropyLoss()
        self.msg_printer = wasabi.Printer()

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
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

        tokens = iter_dict["tokens"]
        instance = iter_dict["instance"]  # List[str]
        instance = instance if isinstance(instance, list) else [instance]
        instance = list(map(lambda instance_str: instance_str.split(), instance))

        encoding = self.elmo_lstm_encoder(x=tokens, instances=instance)

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
