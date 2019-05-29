from typing import Dict, List, Tuple
from parsect.vocab.vocab import Vocab


class Numericalizer:
    """
    This class is for numericalization of instances
    An instance is a List[List[str]] that is tokenized and
    pre-processed. This converts the instances into numerical values that can
    be used in modeling
    """

    def __init__(self, vocabulary: Vocab):
        """
        :param vocabulary: type: Vocab
        Vocab object that is used to build vocab from instances
        """
        self.vocabulary = vocabulary

        if not self.vocabulary.vocab:
            self.vocabulary.build_vocab()

    def numericalize_instance(self, instance: List[str]) -> List[int]:
        """
        Takes an instance List[str] and returns the numericalized version of it
        The `self.max_length` constraint is obeyed.
        If the strings of shorter length, then they are padded
        Else they are truncated to max length
        :param instance:
        :return:
        """
        numerical_tokens = []
        len_tokens = len(instance)

        for string in instance:
            idx = self.vocabulary.get_idx_from_token(string)
            numerical_tokens.append(idx)

        assert len(numerical_tokens) == len_tokens

        return numerical_tokens

    def numericalize_batch_instances(
        self, instances: List[List[str]]
    ) -> List[List[int]]:
        numerical_tokens_batch = []
        for instance in instances:
            tokens = self.numericalize_instance(instance)
            numerical_tokens_batch.append(tokens)

        return numerical_tokens_batch
