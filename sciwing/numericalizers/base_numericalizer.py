from sciwing.vocab.vocab import Vocab
from typing import List
from abc import ABCMeta, abstractmethod
import torch


class BaseNumericalizer(metaclass=ABCMeta):
    def __init__(self, vocabulary: Vocab = None):
        pass

    @abstractmethod
    def numericalize_instance(self, instance: List[str]):
        pass

    @abstractmethod
    def numericalize_batch_instances(self, instances: List[List[str]]):
        pass

    @abstractmethod
    def pad_instance(
        self,
        numericalized_text: List[int],
        max_length: int,
        add_start_end_token: bool = True,
    ):
        pass

    @abstractmethod
    def pad_batch_instances(
        self,
        instances: List[List[int]],
        max_length: int,
        add_start_end_token: bool = True,
    ):
        pass

    @abstractmethod
    def get_mask_for_instance(self, instance: List[int]) -> torch.ByteTensor:
        pass

    @abstractmethod
    def get_mask_for_batch_instances(
        self, instances: List[List[int]]
    ) -> torch.ByteTensor:
        pass
