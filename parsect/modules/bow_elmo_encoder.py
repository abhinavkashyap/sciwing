import torch
import torch.nn as nn
from allennlp.commands.elmo import ElmoEmbedder
from parsect.utils.common import pack_to_length
import wasabi
from typing import List, Iterable


class BowElmoEncoder:
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
        word_aggregation: str = "sum"
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
        :param word_aggregation: type: str
        sum - sum the embeddings across words to obtain sentence embedding
        average - average the embeddings across words to obtain sentence embedding
        """
        super(BowElmoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.layer_aggregation_type = layer_aggregation
        self.word_aggregation_type = word_aggregation
        self.allowed_layer_aggregation_types = ["sum", "average", "last", "first"]
        self.msg_printer = wasabi.Printer()

        assert (
                self.layer_aggregation_type in self.allowed_layer_aggregation_types
        ), self.msg_printer.fail(
            f"For bag of words elmo encoder, the allowable aggregation "
            f"types are {self.allowed_layer_aggregation_types}. You passed {self.layer_aggregation_type}"
        )

        # load the elmo embedders
        with self.msg_printer.loading("Creating Elmo object"):
            self.elmo = ElmoEmbedder()
        self.msg_printer.good("Finished Loading Elmo object")

    def forward(self, x: Iterable[List[str]]) -> torch.Tensor:
        """
        :param x - The input should be a list of instances
        The words are tokenized
        """

        lens_in_batch = [len(instance) for instance in x]
        max_len_in_batch = sorted(lens_in_batch, reverse=True)[0]

        padded_instances = []
        for instance in x:
            padded_instance = pack_to_length(
                tokenized_text=instance, max_length=max_len_in_batch
            )
            padded_instances.append(padded_instance)

        # [np.array] - A generator of embeddings
        # each array in the list is of the shape (3, #words_in_sentence, 1024)
        embedded = list(self.elmo.embed_sentences(padded_instances))

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

        if self.word_aggregation_type == "sum":
            embedding_ = torch.sum(embedding_, dim=1)

        elif self.word_aggregation_type == "average":
            embedding_ = torch.mean(embedding_, dim=1)

        return embedding_

    def __call__(self, x: List[str]):
        return self.forward(x)
