import torch
from allennlp.commands.elmo import ElmoEmbedder
import wasabi
from typing import List, Iterable, Dict, Any
import torch.nn as nn
from parsect.utils.class_nursery import ClassNursery


class BowElmoEmbedder(nn.Module, ClassNursery):
    """
    This trains a non trainable bi-LSTM Elmo Model
    Takes only the last layer of inputs from Elmo
    The representations of all the words are either summed or averaged
    Note that this is not a nn.Module and it has no trainable parameters
    However, the interface is maintained for ease of usability
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        dropout_value: float = 0.0,
        layer_aggregation: str = "sum",
        cuda_device_id: int = -1,
    ):
        """

        :param emb_dim: type: int
        The embedding dimension that is used
        This is fixed in the case of Elmo
        :param dropout_value: type: float
        You can add dropout to the embedding layer
        :param layer_aggregation: type: str
        sum - sums all the layers of elmo embeddings
        average - average all layers of elmo embedding
        first - gets the first layer of embeddings only
        last - gets only last layer of embeddings
        :param cuda_device_id: type: int
        Cuda device that is used to run the model
        """
        super(BowElmoEmbedder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.layer_aggregation_type = layer_aggregation
        self.allowed_layer_aggregation_types = ["sum", "average", "last", "first"]
        self.cuda_device_id = cuda_device_id
        self.msg_printer = wasabi.Printer()

        assert (
            self.layer_aggregation_type in self.allowed_layer_aggregation_types
        ), self.msg_printer.fail(
            f"For bag of words elmo encoder, the allowable aggregation "
            f"types are {self.allowed_layer_aggregation_types}. You passed {self.layer_aggregation_type}"
        )

        # load the elmo embedders
        with self.msg_printer.loading("Creating Elmo object"):
            self.elmo = ElmoEmbedder(cuda_device=self.cuda_device_id)
        self.msg_printer.good("Finished Loading Elmo object")

    def forward(self, iter_dict: Dict[str, Any]) -> torch.Tensor:
        # [np.array] - A generator of embeddings
        # each array in the list is of the shape (3, #words_in_sentence, 1024)
        x = iter_dict["instance"]
        x = [instance.split() for instance in x]

        embedded = list(self.elmo.embed_sentences(x))

        # bs, 3, #words_in_sentence, 1024
        embedded = torch.FloatTensor(embedded)

        embedding_ = None
        # aggregate of word embeddings
        if self.layer_aggregation_type == "sum":
            # bs, #words_in_sentence, 1024
            embedding_ = torch.sum(embedded, dim=1)

        elif self.layer_aggregation_type == "average":
            # mean across all layers
            embedding_ = torch.mean(embedded, dim=1)

        elif self.layer_aggregation_type == "last":
            # bs, max_len, 1024
            embedding_ = embedded[:, -1, :, :]

        elif self.layer_aggregation_type == "first":
            # bs, max_len, 1024
            embedding_ = embedded[:, 0, :, :]

        return embedding_

    def __call__(self, iter_dict: Dict[str, Any]):
        return self.forward(iter_dict)
