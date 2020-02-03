from typing import List, Dict, Union
from sciwing.data.token import Token
from collections import defaultdict


class SeqLabel:
    def __init__(self, labels: List[str], namespace="seq_label"):
        """ Sequential Labels are used to label every token in a line. This mostly gets used for sequential

        Parameters
        ----------
        labels : List[str]
            A list of labels
        namespace : str
            The namespace used for this label
        """
        self.labels = labels
        self.namespace = namespace
        self.tokens: Dict[str, List[Token]] = defaultdict(list)

        for label in labels:
            self.add_token(token=label, namespace=self.namespace)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def namespace(self):
        return self._namespace

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
