from typing import List
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    def __init__(self):
        super(CharacterTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[str]:
        return list(text)

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        batch_tokenized = map(self.tokenize, texts)
        batch_tokenized = list(batch_tokenized)
        return batch_tokenized
