import torch
import torch.nn as nn
from wasabi import Printer
from typing import List, Union
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery


class BOW_Encoder(nn.Module, ClassNursery):
    def __init__(
        self,
        embedder=None,
        dropout_value: float = 0,
        aggregation_type="sum",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """Bag of Words Encoder

        Parameters
        ----------
        embedder : nn.Module
            Any embedder that you would want to use
        dropout_value : float
            The input dropout value that you would want to use
        aggregation_type : str
            The strategy for aggregating words
                sum
                    Aggregate word embedding by summing them
                average
                    Aggregate word embedding by averaging them
        device: Union[torch.device, str]
            The device where the embeddings are stored
        """
        super(BOW_Encoder, self).__init__()
        self.emb_dim = embedder.get_embedding_dimension()
        self.embedder = embedder
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.valid_aggregation_types = ["sum", "average"]
        self.msg_printer = Printer()
        self.device = torch.device(device) if isinstance(device, str) else device

        assert self.aggregation_type in self.valid_aggregation_types

        self.dropout = nn.Dropout(p=self.dropout_value)

    def forward(self, lines: List[Line]) -> torch.FloatTensor:
        """

        Parameters
        ----------
        lines : Dict[str, Any]
            The iter_dict returned by a dataset

        Returns
        -------
        torch.FloatTensor
            The bag of words encoded embedding either average or summed
            The size is [batch_size, embedding_dimension]

        """

        # N * T * D
        embeddings = self.embedder(lines)

        # N * T * D
        embeddings = self.dropout(embeddings)

        if self.aggregation_type == "sum":
            embeddings = torch.sum(embeddings, dim=1)

        elif self.aggregation_type == "average":
            embeddings = torch.mean(embeddings, dim=1)

        return embeddings
