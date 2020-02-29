from typing import List
from sciwing.vocab.vocab import Vocab
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
import torch


class Numericalizer(BaseNumericalizer):
    def __init__(self, vocabulary: Vocab = None):
        """ Numericalizer converts tokens that are strings to numbers

        Parameters
        ----------
        vocabulary : Vocab
            A vocabulary object that is built using a set of tokenized strings

        """
        super().__init__(vocabulary)
        self.vocabulary = vocabulary

        if vocabulary and not self.vocabulary.vocab:
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

    def pad_batch_instances(
        self,
        instances: List[List[int]],
        max_length: int,
        add_start_end_token: bool = True,
    ) -> List[List[int]]:
        """ Pads a batch of instances according to the vocab object

        Parameters
        ----------
        instances : List[List[int]]
        max_length : int
        add_start_end_token : int

        Returns
        -------
        List[List[int]]
        """
        padded_instances = []
        for instance in instances:
            padded_instance = self.pad_instance(
                numericalized_text=instance,
                max_length=max_length,
                add_start_end_token=add_start_end_token,
            )
            padded_instances.append(padded_instance)
        return padded_instances

    @property
    def vocabulary(self):
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, value):
        self._vocabulary = value

    def get_mask_for_instance(self, instance: List[int]) -> torch.BoolTensor:
        start_token_idx = self.vocabulary.get_idx_from_token(
            self.vocabulary.start_token
        )
        end_token_idx = self.vocabulary.get_idx_from_token(self.vocabulary.end_token)
        pad_token_idx = self.vocabulary.get_idx_from_token(self.vocabulary.pad_token)
        unk_token_idx = self.vocabulary.get_idx_from_token(self.vocabulary.unk_token)
        masked_tokens = [start_token_idx, end_token_idx, pad_token_idx, unk_token_idx]

        assert len(set(masked_tokens)) == 4
        mask = [1 if token in masked_tokens else 0 for token in instance]
        mask = torch.BoolTensor(mask)
        return mask

    def get_mask_for_batch_instances(
        self, instances: List[List[int]]
    ) -> torch.BoolTensor:
        masks = []
        for instance in instances:
            mask = self.get_mask_for_instance(instance=instance).tolist()
            masks.append(mask)

        masks = torch.BoolTensor(masks)
        return masks
