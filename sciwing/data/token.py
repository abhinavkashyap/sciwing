from typing import Dict
import torch


class Token:
    def __init__(self, text: str):
        self.text = text
        self.sub_tokens = []

        # a token can hold different kinds of embeddings
        # this is a mapping from the embedding_type to the embedding itself
        self._embedding: Dict[str, torch.FloatTensor] = {}

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def len(self):
        return len(self.text)

    def set_embedding(self, name: str, value: torch.FloatTensor):
        self._embedding[name] = value

    def get_embedding(self, name: str):
        return self._embedding[name]

    @property
    def sub_tokens(self):
        return self._subtokens

    @sub_tokens.setter
    def sub_tokens(self, value):
        self._subtokens = value
