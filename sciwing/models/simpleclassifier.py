import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from typing import List, Any, Dict, Union
from sciwing.data.line import Line
from sciwing.data.label import Label
from wasabi import Printer
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager
import torch


class SimpleClassifier(nn.Module, ClassNursery):
    def __init__(
        self,
        encoder: nn.Module,
        encoding_dim: int,
        num_classes: int,
        classification_layer_bias: bool = True,
        label_namespace: str = "label",
        datasets_manager: DatasetsManager = None,
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ SimpleClassifier is a linear classifier head on top of any encoder

        Parameters
        ----------
        encoder : nn.Module
            Any encoder that takes in lines and produces a single vector
            for every line.
        encoding_dim : int
            The encoding dimension
        num_classes : int
            The number of classes
        classification_layer_bias : bool
            Whether to add classification layer bias or no
            This is set to false only for debugging purposes ff
        label_namespace : str
            The namespace used for labels in the dataset
        datasets_manager: DatasetsManager
            The datasets manager for the model
        device: torch.device
            The device on which the model is run
        """
        super(SimpleClassifier, self).__init__()
        self.encoder = encoder
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.classification_layer_bias = classification_layer_bias
        self.classification_layer = nn.Linear(
            self.encoding_dim, num_classes, bias=self.classification_layer_bias
        )
        self._loss = CrossEntropyLoss()
        self.label_namespace = label_namespace
        self.datasets_manager = datasets_manager
        self.label_numericalizer = self.datasets_manager.namespace_to_numericalizer[
            self.label_namespace
        ]
        self.device = torch.device(device) if isinstance(device, str) else device
        self.msg_printer = Printer()

    def forward(
        self,
        lines: List[Line],
        labels: List[Label] = None,
        is_training: bool = False,
        is_validation: bool = False,
        is_test: bool = False,
    ) -> Dict[str, Any]:
        """

        Parameters
        ----------
        lines : List[Line]
            ``iter_dict`` from any dataset that will be passed on to the encoder
        labels: List[Label]
            A list of labels for every instance
        is_training : bool
            running forward on training dataset?
        is_validation : bool
            running forward on validation dataset?
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

        encoding = self.encoder(lines)

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
            label_indices = []
            for label in labels:
                label_ = label.tokens[self.label_namespace]
                label_ = [tok.text for tok in label_]
                label_ = self.label_numericalizer.numericalize_instance(instance=label_)
                label_indices.append(label_[0])  # taking only the first label here

            labels_tensor = torch.tensor(
                label_indices, device=self.device, dtype=torch.long
            )

            assert labels_tensor.ndimension() == 1, self.msg_printer.fail(
                "the labels should have 1 dimension "
                "your input has shape {0}".format(labels_tensor.size())
            )
            loss = self._loss(logits, labels_tensor)
            output_dict["loss"] = loss

        return output_dict
