from typing import Dict, Union
import wasabi
import numpy as np


class CharEmbLoader:
    def __init__(
        self,
        token2idx: Dict,
        embedding_type: Union[str, None] = None,
        embedding_dimension: Union[str, None] = None,
    ):
        self.token2idx = token2idx
        self.embedding_type = embedding_type
        self.embedding_dimension = embedding_dimension
        self.msg_printer = wasabi.Printer()
        self.vocab_embedding = self.load_embedding()

    def load_embedding(self):
        embedding = {}
        with self.msg_printer.loading("Loading character embedding"):
            tokens = self.token2idx.keys()
            for token in tokens:
                embedding[token] = np.random.normal(
                    loc=0.1, scale=0.1, size=self.embedding_dimension
                )
        self.msg_printer.good("Finished Loading character embedding")
        return embedding
