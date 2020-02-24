from typing import List, Dict, Union
from sciwing.data.token import Token
from collections import defaultdict


class SeqLabel:
    def __init__(self, labels: Dict[str, List[str]]):
        """ Sequential Labels are used to label every token in a line. This mostly gets used for sequential

        Parameters
        ----------
        labels : Dict[str, List[str]]
            A mapping between a namespace and a list of strings
        """
        self.labels = labels
        self.namespace = list(self.labels.keys())
        self.tokens: Dict[str, List[Token]] = defaultdict(list)

        for namespace, label in labels.items():
            self.add_tokens(tokens=label, namespace=namespace)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def namespace(self):
        return list(self.tokens.keys())

    @namespace.setter
    def namespace(self, value):
        self._namespace = value

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    def add_token(self, token: Union[str, Token], namespace=namespace):
        if isinstance(token, str):
            token = Token(token)

        self.tokens[namespace].append(token)

    def add_tokens(self, tokens: Union[List[str], List[Token]], namespace: str):
        for token in tokens:
            self.add_token(token, namespace=namespace)
