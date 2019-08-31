import torch
import torch.nn as nn
from typing import Dict, Any, List
from sciwing.utils.class_nursery import ClassNursery


class ConcatEmbedders(nn.Module, ClassNursery):
    def __init__(self, embedders: List[nn.Module]):
        """ Concatenates a set of embedders into a single embedder.

        Parameters
        ----------
        embedders : List[nn.Module]
            A list of embedders that can be concatenated
        """
        super(ConcatEmbedders, self).__init__()
        self.embedders = embedders

        for idx, embedder in enumerate(self.embedders):
            self.add_module(f"embedder {idx}", embedder)

    def forward(self, iter_dict: Dict[str, Any]):
        """

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            The ``iter_dict`` from any dataset. All the ``keys`` that are expected
            by different embedders are expected to be present in the iterdict

        Returns
        -------
        torch.FloatTensor
            Returns the concatenated embedding that is of the size
            ``[batch_size, time_steps, embedding_dimension]`` where the
            ``embedding_dimension`` is after the concatenation

        """
        embeddings = []
        for embedder in self.embedders:
            embedding = embedder(iter_dict)
            embeddings.append(embedding)

        concat_embedding = torch.cat(embeddings, dim=2)
        return concat_embedding
