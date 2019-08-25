import torch.nn as nn
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
import wasabi
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import Dict, Any
import torch


class ElmoLSTMClassifier(nn.Module):
    def __init__(
        self,
        elmo_lstm_encoder: ElmoLSTMEncoder,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """

        :param elmo_lstm_encoder: type: ElmoLSTMEncoder
        ElmoLSTMEncoder combines the embedding from ElMO BI-LSTM language
        model and a normal token level embedding.
        :param encoding_dim: type: int
        :param num_classes: type: int
        :param classification_layer_bias: type: bool
        Would you want to add bias to the classification layer? set it to true
        This is used for debugging purposes. Use it sparingly
        :param device: torch.device
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
