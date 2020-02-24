import torch
import torch.nn as nn
from typing import List
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.modules.embedders.base_embedders import BaseEmbedder


class ConcatEmbedders(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self, embedders: List[nn.Module], datasets_manager: DatasetsManager = None
    ):
        """ Concatenates a set of embedders into a single embedder.

        Parameters
        ----------
        embedders : List[nn.Module]
            A list of embedders that can be concatenated
        """
        super(ConcatEmbedders, self).__init__()
        self.embedders = embedders
        self.datasets_manager = datasets_manager

        for idx, embedder in enumerate(self.embedders):
            self.add_module(f"embedder_{embedder.embedder_name}", embedder)

    def forward(self, lines: List[Line]):
        """

        Parameters
        ----------
        lines : List[Line]
           A list of Lines.

        Returns
        -------
        torch.FloatTensor
            Returns the concatenated embedding that is of the size
            ``[batch_size, time_steps, embedding_dimension]`` where the
            ``embedding_dimension`` is after the concatenation

        """
        embeddings = []
        for embedder in self.embedders:
            embedding = embedder(lines)
            embeddings.append(embedding)

        concat_embedding = torch.cat(embeddings, dim=2)
        return concat_embedding

    def get_embedding_dimension(self):
        dims = [embedder.get_embedding_dimension() for embedder in self.embedders]
        emb_dim = sum(dims)
        return emb_dim
