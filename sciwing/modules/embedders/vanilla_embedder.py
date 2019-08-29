import torch.nn as nn
from typing import Dict, Any
from sciwing.utils.class_nursery import ClassNursery


class VanillaEmbedder(nn.Module):
    def __init__(self, embedding: nn.Embedding, embedding_dim: int):
        super(VanillaEmbedder, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim

    def forward(self, iter_dict: Dict[str, Any]):
        try:
            tokens = iter_dict["tokens"]  # N * T
            assert tokens.dim() == 2
            embedding = self.embedding(tokens)  # N * T * D
            return embedding
        except AttributeError:
            raise ValueError(f"iter_dict passed should have tokens in them")
        except AssertionError:
            raise ValueError(
                f"tokens passed to vanilla embedder must be 2 dimensions. "
                f"You passed tokens having {tokens.dim()}"
            )
