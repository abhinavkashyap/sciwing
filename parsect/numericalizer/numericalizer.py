from typing import Dict, List, Tuple
from parsect.vocab.vocab import Vocab


class Numericalizer:
    """
    This class is for numericalization of instances
    An instance is a List[List[str]] that is tokenized and
    pre-processed. This converts the instances into numerical values that can
    be used in modeling
    """
    def __init__(self,
                 max_length: int,
                 vocabulary: Vocab):
        """
        :param max_length: type: int
        The maximum length of numericalization
        :param vocabulary: type: Vocab
        Vocab object that is used to build vocab from instances
        """
        self.max_length = max_length
        self.vocabulary = vocabulary

        if not self.vocabulary.vocab:
            self.vocabulary.build_vocab()

    def numericalize_instance(self, instance: List[str]) -> (int, List[int]):
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
        pad_idx = self.vocabulary.get_idx_from_token(self.vocabulary.pad_token)
        start_idx = self.vocabulary.get_idx_from_token(self.vocabulary.start_token)
        end_idx = self.vocabulary.get_idx_from_token(self.vocabulary.end_token)

        for string in instance:
            idx = self.vocabulary.get_idx_from_token(string)
            numerical_tokens.append(idx)

        # pad the output to max length
        if len_tokens < self.max_length:
            len_difference = self.max_length - len_tokens
            len_difference = len_difference - 2

            # add start and end before padding
            numerical_tokens.insert(0, start_idx)
            numerical_tokens.append(end_idx)

            # pad to max_length
            numerical_tokens.extend([pad_idx] * len_difference)

        else:
            # allow space for <SOS> and <EOS>
            numerical_tokens = numerical_tokens[:self.max_length-2]
            numerical_tokens.insert(0, start_idx)
            numerical_tokens.append(end_idx)

        assert len(numerical_tokens) == self.max_length

        return len_tokens, numerical_tokens

    def numericalize_batch_instances(self, instances: List[List[str]]) -> (
            List[int], List[List[int]]):
        len_numerical_tokens_batch = map(self.numericalize_instance, instances)
        lengths_batch = []
        numerical_tokens_batch = []
        for length, numerical_tokens in len_numerical_tokens_batch:
            lengths_batch.append(length)
            numerical_tokens_batch.append(numerical_tokens)

        return lengths_batch, numerical_tokens_batch

