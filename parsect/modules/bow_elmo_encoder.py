import torch
import torch.nn as nn
from allennlp.commands.elmo import ElmoEmbedder
import wasabi
from typing import List


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
        aggregation_type: str = "sum",
    ):
        """

        :param emb_dim: type: int
        The embedding dimension that is used
        This is fixed in the case of Elmo
        :param dropout_value: type: float
        You can add dropout to the embedding layer
        :param aggregation_type: type: str
        sum - sums the embeddings of tokens in an instance
        average - averages the embedding of tokens in an instance
        """
        super(BowElmoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.msg_printer = wasabi.Printer()

        # load the elmo embedders
        with self.msg_printer.loading("Creating Elmo object"):
            self.elmo = ElmoEmbedder()
        self.msg_printer.good("Finished Loading Elmo object")

    def forward(self, x: List[str]) -> torch.Tensor:
        """
        :param x - The input should be a list of instances
        The words are tokenized
        """
        # [np.array] - A generator of embeddings
        # each array in the list is of the shape (3, #words_in_sentence, 1024)
        embedded = list(self.elmo.embed_sentences(x))
        embeddings = []

        for embedding in embedded:
            # num words, 1024
            last_layer_embedding = torch.FloatTensor(embedding[2, :, :])
            embedding_ = None

            # aggregate of word embeddings
            if self.aggregation_type == "sum":
                embedding_ = torch.sum(last_layer_embedding, dim=0)
            elif self.aggregation_type == "average":
                embedding_ = torch.mean(last_layer_embedding, dim=0)

            embeddings.append(embedding_)

        # number of sentences * 1024
        embeddings = torch.stack(embeddings, dim=0)

        return embeddings

    def __call__(self, x: List[str]):
        return self.forward(x)
