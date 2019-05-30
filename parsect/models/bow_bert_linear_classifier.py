import torch.nn as nn
from typing import Dict, Any
from torch.nn import CrossEntropyLoss
import wasabi
from torch.nn.functional import softmax
from parsect.modules.bow_bert_encoder import BowBertEncoder


class BowBertLinearClassifier(nn.Module):
    def __init__(self,
                 encoder: BowBertEncoder,
                 encoding_dim: int,
                 num_classes: int,
                 classification_layer_bias: bool = True):
        super(BowBertLinearClassifier, self).__init__()
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

        # N * D
        # N - batch size
        # D - Encoding dimension `self.encoding_dim`

        labels = iter_dict['label']
        labels = labels.squeeze(1)
        x = iter_dict["raw_instance"]
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
