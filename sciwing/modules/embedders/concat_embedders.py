import torch
import torch.nn as nn
from typing import Dict, Any, List
from sciwing.utils.class_nursery import ClassNursery


class ConcatEmbedders(nn.Module, ClassNursery):
    def __init__(self, embedders: List[nn.Module]):
        super(ConcatEmbedders, self).__init__()
        self.embedders = embedders

        for idx, embedder in enumerate(self.embedders):
            self.add_module(f"embedder {idx}", embedder)

    def forward(self, iter_dict: Dict[str, Any]):
        embeddings = []
        for embedder in self.embedders:
            embedding = embedder(iter_dict)
            embeddings.append(embedding)

        concat_embedding = torch.cat(embeddings, dim=2)
        return concat_embedding
