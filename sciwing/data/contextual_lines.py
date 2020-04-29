from typing import List, Dict, Any, Union
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.data.token import Token
from sciwing.data.line import Line
from collections import defaultdict
from sciwing.tokenizers.word_tokenizer import WordTokenizer


class LineWithContext:
    """
        There are multiple situations where every line is accompanied by other lines
        and is required for making predictions current line. This class encompasses
        other lines. This is a thin wrapper around the lines themselves and
        acts like a container line
    """

    def __init__(
        self, text: str, context: List[str], tokenizers: Dict[str, BaseTokenizer] = None
    ):
        if tokenizers is None:
            tokenizers = {"tokens": WordTokenizer()}
        self.text = text
        self.context = context
        self.tokenizers = tokenizers
        self.tokens: Dict[str, List[Any]] = defaultdict(list)
        self.namespaces = list(tokenizers.keys())
        for namespace in tokenizers.keys():
            self.namespaces.append(f"contextual_{namespace}")

        # add tokens for the word tokens
        for namespace, tokenizer in self.tokenizers.items():
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                self.add_token(token=token, namespace=namespace)

        # add tokens for the contextual lines
        for namespace, tokenizer in self.tokenizers.items():
            for contextual_line in self.context:
                tokens = tokenizer.tokenize(contextual_line)
                tokens = [Token(tok) for tok in tokens]
                self.tokens[f"contextual_{namespace}"].append(tokens)

        self.line = Line(text=text, tokenizers=self.tokenizers)
        self.context_lines = []
        for text in self.context:
            context_line = Line(text=text, tokenizers=self.tokenizers)
            self.context_lines.append(context_line)

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
