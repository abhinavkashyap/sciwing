import torch
import torch.nn as nn
from wasabi import Printer
from typing import Dict, Any
from parsect.utils.class_nursery import ClassNursery


class BOW_Encoder(nn.Module, ClassNursery):
    def __init__(
        self,
        emb_dim: int = 100,
        embedder=None,
        dropout_value: float = 0,
        aggregation_type="sum",
    ):
        super(BOW_Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.embedder = embedder
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.valid_aggregation_types = ["sum", "average"]
        self.msg_printer = Printer()

        assert self.aggregation_type in self.valid_aggregation_types

        self.dropout = nn.Dropout(p=self.dropout_value)

    def forward(self, iter_dict: Dict[str, Any]) -> torch.FloatTensor:

        # N * T * D
        embeddings = self.embedder(iter_dict)

        # N * T * D
        embeddings = self.dropout(embeddings)

        if self.aggregation_type == "sum":
            embeddings = torch.sum(embeddings, dim=1)

        elif self.aggregation_type == "average":
            embeddings = torch.mean(embeddings, dim=1)

        return embeddings
