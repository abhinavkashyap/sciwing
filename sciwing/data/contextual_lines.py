from typing import List, Dict, Union
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.data.token import Token
from collections import defaultdict
from sciwing.data.line import Line


class LinesWithContext(Line):
    """
        There are multiple situations where every line is accompanied by other lines
        and is required for making predictions current line. This class encompasses
        other lines. This extends the Line class but adds the context lines
    """

    def __init__(
        self, text: str, context: List[str], tokenizers: Dict[str, BaseTokenizer] = None
    ):
        super(LinesWithContext, self).__init__(text=text, tokenizers=tokenizers)
        self.context = context
        self.context_tokens: Dict[str, List[List[Token]]] = defaultdict(list)

        for context_text in context:
            for namespace, tokenizer in self.tokenizers.items():
                tokens = tokenizer.tokenize(context_text)
                self.add_context_tokens(tokens=tokens, namespace=namespace)

    def add_context_tokens(self, tokens: Union[List[str], List[Token]], namespace: str):
        if isinstance(tokens[0], Token):
            tokens = list(map(lambda tok: Token(tok), tokens))

        self.context_tokens[namespace].append(tokens)

    def get_context_tokens(self, namespace: str):
        return self.context_tokens[namespace]
