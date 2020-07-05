import os
import sciwing.constants as constants
from typing import Dict, Union
import numpy as np
from tqdm import tqdm
from wasabi import Printer
import gensim
from sciwing.vocab.vocab import Vocab
import torch
import pathlib
from sciwing.utils.common import cached_path


PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]
EMBEDDING_FILE_URLS = constants.EMBEDDING_FILE_URLS


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
        self.embedding_type = embedding_type
        self.embedding_cache_dir = pathlib.Path(EMBEDDING_CACHE_DIR)

        if not self.embedding_cache_dir.is_dir():
            self.embedding_cache_dir.mkdir(parents=True)

        self.allowed_embedding_types = [
            "glove_6B_50",
            "glove_6B_100",
            "glove_6B_200",
            "glove_6B_300",
            "parscit",
            "lample_conll",
        ]

        assert self.embedding_type in self.allowed_embedding_types, (
            f"You can use one of {self.allowed_embedding_types} for embedding type."
            f"You passed {self.embedding_type}"
        )
        self.embedding_filename = self.get_preloaded_filename()
        self.vocab_embedding = {}  # stores the embedding for all words in vocab
        self.msg_printer = Printer()
        self._embeddings: Dict[str, np.array] = {}

        if "glove" in self.embedding_type:
            self._embeddings = self.load_glove_embedding()

        if "parscit" in self.embedding_type:
            self._embeddings = self.load_parscit_embedding()

        if self.embedding_type == "lample_conll":
            self._embeddings = self.load_lample_conll_embedding()

    def get_preloaded_filename(self):
        filename = None
        url = None

        if self.embedding_type == "glove_6B_50":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.50d.txt")
            url = EMBEDDING_FILE_URLS["GLOVE_FILE"]

        elif self.embedding_type == "glove_6B_100":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.100d.txt")
            url = EMBEDDING_FILE_URLS["GLOVE_FILE"]

        elif self.embedding_type == "glove_6B_200":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.200d.txt")
            url = EMBEDDING_FILE_URLS["GLOVE_FILE"]

        elif self.embedding_type == "glove_6B_300":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "glove.6B.300d.txt")
            url = EMBEDDING_FILE_URLS["GLOVE_FILE"]

        elif self.embedding_type == "parscit":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "vectors_with_unk.kv")
            url = EMBEDDING_FILE_URLS["PARSCIT_EMBEDDINGS"]

        elif self.embedding_type == "lample_conll":
            filename = os.path.join(EMBEDDING_CACHE_DIR, "lample_conll")
            url = EMBEDDING_FILE_URLS["LAMPLE_CONLL"]
        else:
            raise ValueError(
                f"Check the embedding type. It has to be one of {self.allowed_embedding_types}"
            )

        url_path = pathlib.Path(url)
        destination_path = url_path.parts[-1]
        destination_path = self.embedding_cache_dir.joinpath(destination_path)
        _ = cached_path(url=url, unzip=True, path=destination_path)

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

    def load_lample_conll_embedding(self) -> Dict[str, np.array]:
        embedding_dim = 100
        self.embedding_dimension = embedding_dim
        lample_conll_embedding: Dict[str, np.array] = {}
        with open(self.embedding_filename, "r") as fp:
            for line in tqdm(
                fp,
                desc=f"Loading Lample CoNLL embedding from file {self.embedding_filename}",
            ):
                values = line.split()
                word = values[0]
                embedding = values[1:]
                embedding = list(map(lambda value: float(value), embedding))
                embedding = np.array(embedding)
                lample_conll_embedding[word] = embedding
        return lample_conll_embedding

    def get_embeddings_for_vocab(self, vocab: Vocab) -> torch.FloatTensor:
        idx2item = vocab.get_idx2token_mapping()
        len_vocab = len(idx2item)
        embeddings = []
        for idx in range(len_vocab):
            item = idx2item.get(idx)
            try:
                # try getting the embeddings from the embeddings dictionary
                emb = self._embeddings[item]
            except KeyError:
                try:
                    # try lowercasing the item and getting the embedding
                    emb = self._embeddings[item.lower()]
                except KeyError:
                    # nothing is working, lets fill it with random integers from normal dist
                    emb = np.random.randn(self.embedding_dimension)
            embeddings.append(emb)

        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value
