from typing import List
from sciwing.vocab.vocab import Vocab
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer


class NumericalizerForTransformer(BaseNumericalizer):
    def __init__(self, vocab: Vocab = None, tokenizer: TokenizerForBert = None):
        super(NumericalizerForTransformer, self).__init__()
        self.vocab = vocab
        self.tokenizer = tokenizer

    def numericalize_instance(self, instance: List[str]) -> List[int]:
        tokens = self.tokenizer.tokenizer.convert_tokens_to_ids(instance)
        return tokens

    def numericalize_batch_instances(self, instances: List[List[str]]) -> List[int]:
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
        start_token_idx = self.tokenizer.tokenizer.vocab["[CLS]"]
        end_token_idx = self.tokenizer.tokenizer.vocab["[SEP]"]
        pad_token_idx = self.tokenizer.tokenizer.vocab["[PAD]"]

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
    ):
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
