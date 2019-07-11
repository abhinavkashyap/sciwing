from abc import ABCMeta, abstractmethod
from typing import List


class BaseTokenizer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        pass
