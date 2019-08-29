from typing import Dict, List, Tuple
from sciwing.vocab.vocab import Vocab


class Numericalizer:
    def __init__(self, vocabulary: Vocab):
        self.vocabulary = vocabulary

        if not self.vocabulary.vocab:
            self.vocabulary.build_vocab()

    def numericalize_instance(self, instance: List[str]) -> List[int]:
        """ Numericalizes instances
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
