import torch.nn as nn
import torch
from typing import List, Union
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.vocab.embedding_loader import EmbeddingLoader
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.data.datasets_manager import DatasetsManager


class WordEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        embedding_type: str,
        datasets_manager: DatasetsManager = None,
        word_tokens_namespace="tokens",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Word Embedder embeds the tokens using the desired embeddings. These are static
        embeddings.

        Parameters
        ----------
        embedding_type : str
            The type of embedding that you would want
        datasets_manager: DatasetsManager
            The datasets manager which is running your experiments
        word_tokens_namespace: str
            The namespace where the word tokens are stored in your data
        device: Union[torch.device, str]
            The device on which this embedder is run
        """
        super(WordEmbedder, self).__init__()

        self.embedding_type = embedding_type
        self.datasets_manager = datasets_manager
        self.embedding_loader = EmbeddingLoader(embedding_type=self.embedding_type)
        self.embedder_name = embedding_type
        self.embedding_dimension = self.get_embedding_dimension()
        self.word_tokens_namespace = word_tokens_namespace
        self.device = torch.device(device) if isinstance(device, str) else device

    def forward(self, lines: List[Line]) -> torch.FloatTensor:
        """ This will only consider the "tokens" present in the line. The namespace
        for the tokens is set with the class instantiation

        Parameters
        ----------
        lines : List[Line]


        Returns
        -------
        torch.FloatTensor
            It returns the embedding of the size ``[batch_size, max_num_timesteps, embedding_dimension]``

        """

        for line in lines:
            for token in line.tokens[self.word_tokens_namespace]:
                try:
                    emb = self.embedding_loader.embeddings[token.text]
                    emb = torch.tensor(emb, dtype=torch.float, device=self.device)
                except KeyError:
                    try:
                        emb = self.embedding_loader.embeddings[token.text.lower()]
                        emb = torch.tensor(emb, dtype=torch.float, device=self.device)
                    except KeyError:
                        emb = torch.zeros(
                            self.embedding_dimension,
                            device=self.device,
                            dtype=torch.float,
                        )

                token.set_embedding(name=self.embedder_name, value=emb)

        # return the [batch_size, longest_sequence, embedding_dimension]
        # This module store all the information in the tokens and the sentences

        line_lengths = [len(line.tokens[self.word_tokens_namespace]) for line in lines]
        max_line_length = max(line_lengths)

        batch_embeddings = []
        for idx, length in enumerate(line_lengths):
            sentence_embedding = []
            padding_length = max_line_length - length
            line = lines[idx]
            tokens = line.tokens[self.word_tokens_namespace]
            for token in tokens:
                token_embedding = token.get_embedding(name=self.embedder_name)
                sentence_embedding.append(token_embedding)
            for i in range(padding_length):
                zeros = torch.randn(
                    self.embedding_loader.embedding_dimension,
                    device=self.device,
                    dtype=torch.float,
                )
                sentence_embedding.append(zeros)

            sentence_embedding = torch.stack(sentence_embedding)
            batch_embeddings.append(sentence_embedding)

        batch_embeddings = torch.stack(batch_embeddings)
        return batch_embeddings

    def get_embedding_dimension(self) -> int:
        return self.embedding_loader.embedding_dimension
