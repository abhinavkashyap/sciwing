import torch.nn as nn
from typing import Dict, Any


class VanillaEmbedder(nn.Module):
    def __init__(self, embedding: nn.Embedding, embedding_dim: int):
        super(VanillaEmbedder, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim

    def forward(self, iter_dict: Dict[str, Any]):
        try:
            tokens = iter_dict["tokens"]  # N * T
        except AttributeError:
            raise ValueError(f"iter_dict passed should have tokens in them")
        embedding = self.embedding(tokens)  # N * T * D
        return embedding
