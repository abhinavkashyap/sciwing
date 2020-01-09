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

    def pad_instance(
        self,
        numericalized_text: List[int],
        max_length: int,
        add_start_end_token: bool = True,
    ) -> List[int]:
        """ Pads the instance according to the vocab object

        Parameters
        ----------
        numericalized_text : List[int]
            Pads a numericalized instance
        max_length: int
            The maximum length to pad to
        add_start_end_token: bool
            If true, start and end token will be added to
            the tokenized text

        Returns
        -------
        List[int]
            Padded instance

        """
        start_token_idx = self.vocabulary.get_idx_from_token(
            self.vocabulary.start_token
        )
        end_token_idx = self.vocabulary.get_idx_from_token(self.vocabulary.end_token)
        pad_token_idx = self.vocabulary.get_idx_from_token(self.vocabulary.pad_token)

        if not add_start_end_token:
            numericalized_text = numericalized_text[:max_length]
        else:
            max_length = max_length if max_length > 2 else 2
            numericalized_text = numericalized_text[: max_length - 2]
            numericalized_text.append(end_token_idx)
            numericalized_text.insert(0, start_token_idx)

        pad_length = max_length - len(numericalized_text)
        for i in range(pad_length):
            numericalized_text.append(pad_token_idx)

        assert len(numericalized_text) == max_length

        return numericalized_text

    def pad_batch_instances(self, instances: List[List[int]]) -> List[List[int]]:
        """ Pads a batch of instances according to the vocab object

        Parameters
        ----------
        instances : List[List[int]]

        Returns
        -------
        List[List[int]]
        """
        padded_instances = map(self.pad_instance, instances)
        padded_instances = list(padded_instances)
        return padded_instances
