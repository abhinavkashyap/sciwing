from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.vocab.embedding_loader import EmbeddingLoader
import torch
import torch.nn as nn
from typing import List
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery


class TrainableWordEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        embedding_type: str,
        datasets_manager: DatasetsManager = None,
        word_tokens_namespace: str = "tokens",
        device: torch.device = torch.device("cpu"),
    ):
        """
        This represents trainable word embeddings which are trained along with the parameters
        of the network. The embeddings in the class `WordEmbedder` are not trainable. They are
        static

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
        super(TrainableWordEmbedder, self).__init__()
        self.embedding_type = embedding_type
        self.datasets_manager = datasets_manager
        self.word_tokens_namespace = word_tokens_namespace
        self.device = torch.device(device) if isinstance(device, str) else device
        self.embedding_loader = EmbeddingLoader(embedding_type=embedding_type)
        self.embedder_name = self.embedding_loader.embedding_type
        self.embedding_dimension = self.get_embedding_dimension()
        self.vocab = self.datasets_manager.namespace_to_vocab[
            self.word_tokens_namespace
        ]
        self.numericalizer = self.datasets_manager.namespace_to_numericalizer[
            self.word_tokens_namespace
        ]
        embeddings = self.embedding_loader.get_embeddings_for_vocab(self.vocab)

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embeddings, freeze=False
        )

    def forward(self, lines: List[Line]) -> torch.FloatTensor:
        line_lengths = [len(line.tokens[self.word_tokens_namespace]) for line in lines]
        max_line_length = max(line_lengths)

        numericalized_tokens = []
        for line in lines:
            tokens = line.tokens[self.word_tokens_namespace]
            tokens = [tok.text for tok in tokens]
            tokens = self.numericalizer.numericalize_instance(instance=tokens)
            tokens = self.numericalizer.pad_instance(
                numericalized_text=tokens,
                max_length=max_line_length,
                add_start_end_token=False,
            )
            numericalized_tokens.append(tokens)

        numericalized_tokens = torch.tensor(
            numericalized_tokens, dtype=torch.long, device=self.device
        )
        embedding = self.embedding(numericalized_tokens)
        return embedding

    def get_embedding_dimension(self) -> int:
        return self.embedding_loader.embedding_dimension
