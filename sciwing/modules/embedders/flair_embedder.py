from flair.data import Sentence
from sciwing.data.line import Line
from typing import List, Union
import torch
from flair.embeddings import FlairEmbeddings
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.utils.class_nursery import ClassNursery
from sciwing.modules.embedders.base_embedders import BaseEmbedder
import torch.nn as nn


class FlairEmbedder(nn.Module, ClassNursery, BaseEmbedder):
    def __init__(
        self,
        embedding_type: str,
        datasets_manager: DatasetsManager = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super(FlairEmbedder, self).__init__()
        self.allowed_type = ["en", "news"]
        assert embedding_type in self.allowed_type
        self.embedder_forward = FlairEmbeddings(f"{embedding_type}-forward")
        self.embedder_backward = FlairEmbeddings(f"{embedding_type}-backward")
        self.embedder_name = f"FlairEmbedder-{embedding_type}"
        self.datasets_manager = datasets_manager
        self.device = torch.device(device) if isinstance(device, str) else device

    def forward(self, lines: List[Line]):
        sentences = []
        for line in lines:
            sentence = Sentence(line.text)
            sentences.append(sentence)

        _ = self.embedder_forward.embed(sentences)
        _ = self.embedder_backward.embed(sentences)

        batch_embeddings = []
        for sentence in sentences:
            sentence_embeddings = []
            for token in sentence:
                embedding = token.get_embedding()
                sentence_embeddings.append(embedding)

            batch_embeddings.append(sentence_embeddings)

        batch_embeddings = torch.stack(
            batch_embeddings, dtype=torch.float, device=self.device
        )
        return batch_embeddings

    def get_embedding_dimension(self):
        return self.embedder_forward.embedding_length * 2  # for forward and backward
