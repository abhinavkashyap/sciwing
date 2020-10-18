from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.data.token import Token
from typing import Union, List, Dict, Any
from collections import defaultdict


class SeqSentence:
    def __init__(self, sents: List[str], tokenizers: Dict[str, BaseTokenizer] = None):
        if tokenizers is None:
            tokenizers = {"tokens": WordTokenizer()}
        self.sents = sents
        self.tokenizers = tokenizers
        self.tokens: Dict[str, List[List[Any]]] = defaultdict(list)
        self.namespaces = list(tokenizers.keys())

        for namespace, tokenizer in tokenizers.items():
            for sent in sents:
                sent_tokens = tokenizer.tokenize(sent)
                self.add_sent_tokens(tokens=sent_tokens, namespace=namespace)

    def add_sent_tokens(self, tokens: Union[List[str], List[Token]], namespace: str):
        sent_tokens = []
        for token in tokens:
            if isinstance(token, str):
                token = Token(token)
            sent_tokens.append(token)
        self.tokens[namespace].append(sent_tokens)

    def add_tokens(self, sents: str, tokenizers: Dict[str, BaseTokenizer] = None):
        for namespace, tokenizer in tokenizers.items():
            for sent in sents:
                sent_tokens = tokenizer.tokenize(sent)
                self.add_sent_tokens(tokens=sent_tokens, namespace=namespace)

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
