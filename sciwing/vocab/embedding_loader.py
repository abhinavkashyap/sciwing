import os
import sciwing.constants as constants
from typing import Dict, Union
import numpy as np
from tqdm import tqdm
from wasabi import Printer
import gensim
from sciwing.vocab.vocab import Vocab
import torch


PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class EmbeddingLoader:
    """
    This handles the loading of word embeddings for a vocab
    This can handle different kinds of embeddings.

    """

    def __init__(self, embedding_type: Union[str] = "glove_6B_50"):
        """

        Parameters
        ----------
        embedding_type : str
            The type of embedding that needs to be loaded
        """
        self.embedding_dimension = None
        self.embedding_type = "glove_" if embedding_type is None else embedding_type

        self.allowed_embedding_types = [
            "glove_6B_50",
            "glove_6B_100",
            "glove_6B_200",
            "glove_6B_300",
            "parscit",
        ]

        assert (
            self.embedding_type in self.allowed_embedding_types
        ), f"You can use one of {self.allowed_embedding_types} for embedding type"
        self.embedding_filename = self.get_preloaded_filename()
        self.vocab_embedding = {}  # stores the embedding for all words in vocab
        self.msg_printer = Printer()
        self._embeddings: Dict[str, np.array] = {}

        if "glove" in self.embedding_type:
            self._embeddings = self.load_glove_embedding()

        if "parscit" in self.embedding_type:
            self._embeddings = self.load_parscit_embedding()

    def get_preloaded_filename(self):
        filename = None

        if self.embedding_type == "glove_6B_50":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.50d.txt")

        elif self.embedding_type == "glove_6B_100":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.100d.txt")

        elif self.embedding_type == "glove_6B_200":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.200d.txt")

        elif self.embedding_type == "glove_6B_300":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.300d.txt")
        elif self.embedding_type == "parscit":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "vectors_with_unk.kv")

        return filename

    def load_glove_embedding(self) -> Dict[str, np.array]:
        """
        Imports the glove embedding
        Loads the word embedding for words in the vocabulary
        If the word in the vocabulary doesnot have an embedding
        then it is loaded with zeros
        """
        embedding_dim = int(self.embedding_type.split("_")[-1])
        self.embedding_dimension = embedding_dim
        glove_embeddings: Dict[str, np.array] = {}
        with self.msg_printer.loading("Loading GLOVE embeddings"):
            with open(self.embedding_filename, "r") as fp:
                for line in tqdm(
                    fp,
                    desc="Loading embeddings from file {0}".format(self.embedding_type),
                ):
                    values = line.split()
                    word = values[0]
                    embedding = np.array([float(value) for value in values[1:]])
                    glove_embeddings[word] = embedding

        return glove_embeddings

    def load_parscit_embedding(self) -> Dict[str, np.array]:
        pretrained = gensim.models.KeyedVectors.load(self.embedding_filename, mmap="r")
        self.embedding_dimension = 500
        return pretrained

    def get_embeddings_for_vocab(self, vocab: Vocab) -> torch.FloatTensor:
        idx2item = vocab.get_idx2token_mapping()
        len_vocab = len(idx2item)
        embeddings = []
        for idx in range(len_vocab):
            item = idx2item.get(idx)
            try:
                emb = self._embeddings[item]
            except KeyError:
                try:
                    emb = self._embeddings[item.lower()]
                except KeyError:
                    emb = np.zeros(shape=self.embedding_dimension)
            embeddings.append(emb)

        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value
