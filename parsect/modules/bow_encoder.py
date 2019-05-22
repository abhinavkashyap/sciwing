import torch
import torch.nn as nn
from wasabi import Printer


class BOW_Encoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        embedding: torch.nn.Embedding,
        dropout_value: float = 0,
        aggregation_type="sum",
    ):
        """

        :param emb_dim: type: int
        The embedding dimension of the encoder
        :param embedding: type: torch.nn.Embedding or anything that works similar to it
        The embedding for all the words in the enncoder
        You can pass any embedding here. GloVE, ElMO, BERT, Sci-BERT
        :param dropout_value: type: float
        you can add dropout to the embedding layer
        :param aggregation_type:  type: str
        sum - sums the embeddings of tokens in an instance
        average - averages the embedding of tokens in an instance
        """
        super(BOW_Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = embedding
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.valid_aggregation_types = ["sum", "average"]
        self.msg_printer = Printer()

        assert self.aggregation_type in self.valid_aggregation_types

        self.dropout = nn.Dropout(p=self.dropout_value)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """

        :param x: type: torch.LongTensor
        shape: N * T
        N - Batch size
        T - Number of tokens
        :return:
        """
        assert x.ndimension() == 2, self.msg_printer.fail(
            "The input should be 2 dimennsional, "
            "you passed a {0}-dimensional input".format(x.size())
        )
        # N * T * D
        embeddings = self.embedding(x)

        # N * T * D
        embeddings = self.dropout(embeddings)

        if self.aggregation_type == "sum":
            embeddings = torch.sum(embeddings, dim=1)

        elif self.aggregation_type == "average":
            embeddings = torch.mean(embeddings, dim=1)

        return embeddings
