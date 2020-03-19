from flair.data import Sentence
from sciwing.data.line import Line
from typing import List, Union
import torch
import flair
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
        word_tokens_namespace: str = "tokens",
    ):
        """ Flair Embeddings. This is used to produce Named Entity Recognition. Note: This only
        works if your tokens are produced by splitting based on white space

        Parameters
        ----------
        embedding_type
        datasets_manager
        device
        word_tokens_namespace
        """
        super(FlairEmbedder, self).__init__()
        self.allowed_type = ["en", "news"]
        assert embedding_type in self.allowed_type
        self.embedder_forward = FlairEmbeddings(f"{embedding_type}-forward")
        self.embedder_backward = FlairEmbeddings(f"{embedding_type}-backward")
        self.embedder_name = f"FlairEmbedder-{embedding_type}"
        self.datasets_manager = datasets_manager
        self.device = torch.device(device) if isinstance(device, str) else device
        self.word_tokens_namespace = word_tokens_namespace

    def forward(self, lines: List[Line]):
        sentences = []
        for line in lines:
            sentence = Sentence(line.text)
            sentences.append(sentence)

        len_tokens = [len(line.tokens[self.word_tokens_namespace]) for line in lines]
        max_len = max(len_tokens)

        _ = self.embedder_forward.embed(sentences)
        _ = self.embedder_backward.embed(sentences)

        batch_embeddings = []
        for sentence in sentences:
            sentence_embeddings = []
            padding_length = max_len - len(sentence)
            for token in sentence:
                embedding = token.get_embedding()
                embedding = embedding.to(self.device)
                sentence_embeddings.append(embedding)
            for i in range(padding_length):
                embedding = torch.randn(
                    self.get_embedding_dimension(),
                    dtype=torch.float,
                    device=self.device,
                )
                sentence_embeddings.append(embedding)

            sentence_embeddings = torch.stack(sentence_embeddings)
            batch_embeddings.append(sentence_embeddings)

        # batch_size, num_tokens, embedding_dim
        batch_embeddings = torch.stack(batch_embeddings)
        batch_embeddings = batch_embeddings.to(self.device)

        for idx, line in enumerate(lines):
            line_embeddings = batch_embeddings[idx]
            for token, emb in zip(
                line.tokens[self.word_tokens_namespace], line_embeddings
            ):
                token.set_embedding(name=self.embedder_name, value=emb)

        return batch_embeddings

    def get_embedding_dimension(self):
        return self.embedder_forward.embedding_length * 2  # for forward and backward
