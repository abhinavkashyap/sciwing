import torch.nn as nn
from typing import Dict, Any
from sciwing.utils.class_nursery import ClassNursery


class VanillaEmbedder(nn.Module, ClassNursery):
    def __init__(self, embedding: nn.Embedding, embedding_dim: int):
        """ Vanilla Embedder embeds the tokens using the embedding passed.

        Parameters
        ----------
        embedding : nn.Embedding
            A pytoch embedding that maps textual units to embeddings
        embedding_dim : int
            The embedding dimension
        """
        super(VanillaEmbedder, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim

    def forward(self, iter_dict: Dict[str, Any]):
        """

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` from a dataset. It expects ``tokens`` to be present as a key of the
            iter_dict which is usually of the shape ``[batch_size, max_num_timesteps]``

        Returns
        -------
        torch.FloatTensor
            It returns the embedding of the size ``[batch_size, max_num_timesteps, embedding_dimension]``

        """
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
