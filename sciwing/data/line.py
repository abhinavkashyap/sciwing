from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.data.token import Token
from typing import Union, List, Dict, Any
from collections import defaultdict


class Line:
    def __init__(self, text: str, tokenizers: Dict[str, BaseTokenizer] = None):
        if tokenizers is None:
            tokenizers = {"tokens": WordTokenizer()}
        self.text = text
        self.tokenizers = tokenizers
        self.tokens: Dict[str, List[Any]] = defaultdict(list)
        self.namespaces = list(tokenizers.keys())

        for namespace, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                self.add_token(token=token, namespace=namespace)

    def add_token(self, token: Union[Token, str], namespace: str):
        if isinstance(token, str):
            token = Token(token)

        self.tokens[namespace].append(token)

    def add_tokens(self, tokens: Union[List[str], List[Token]], namespace: str):
        for token in tokens:
            self.add_token(token, namespace=namespace)

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @property
    def namespaces(self):
        return self._namespaces

    @namespaces.setter
    def namespaces(self, value):
        self._namespaces = value
