from sciwing.data.token import Token
from typing import Union, List, Dict
from collections import defaultdict


class Label:
    def __init__(self, text: str, namespace: str = "label"):
        """ Defines a single label for an example
        We will only consider one namespace for this class
        Also we will only consider a single token name for ever label

        label = Label(text="title")

        Parameters
        ----------
        text : str
        namespace : str
        """
        self.text = text
        self.namespace = namespace
        self.tokens: Dict[str, List[Token]] = defaultdict(list)
        self.add_token(token=self.text, namespace=namespace)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

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

    def add_token(self, token: Union[Token, str], namespace: str):
        if isinstance(token, str):
            token = Token(token)

        self.tokens[namespace].append(token)

    def add_tokens(self, tokens: Union[List[str], List[Token]], namespace: str):
        for token in tokens:
            self.add_token(token, namespace=namespace)
