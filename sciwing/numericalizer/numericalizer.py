from typing import Dict, List, Tuple
from sciwing.vocab.vocab import Vocab


class Numericalizer:
    def __init__(self, vocabulary: Vocab):
        """ Numericalizer converts tokens that are strings to numbers

        Parameters
        ----------
        vocabulary : Vocab
            A vocabulary object that is built using a set of tokenized strings

        """
        self.vocabulary = vocabulary

        if not self.vocabulary.vocab:
            self.vocabulary.build_vocab()

    def numericalize_instance(self, instance: List[str]) -> List[int]:
        """ Numericalize a single instance

        Parameters
        ------------
        instance : List[str]
            An instance is a list of tokens


        Returns
        --------------
        List[int]
            Numericalized instance
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
        """ Numericalizes a batch of instances

        Parameters
        ----------
        instances : List[List[str]]
            A list of tokenized sentences

        Returns
        -------
        List[List[int]]
            A list of numericalized instances

        """
        numerical_tokens_batch = []
        for instance in instances:
            tokens = self.numericalize_instance(instance)
            numerical_tokens_batch.append(tokens)

        return numerical_tokens_batch
