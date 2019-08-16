from parsect.datasets.sprinkle_dataset import sprinkle_dataset
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)

from typing import Dict, Any, Optional, Union, List
from parsect.vocab.vocab import Vocab
from parsect.utils.common import pack_to_length
import torch
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.tokenizers.character_tokenizer import CharacterTokenizer
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.utils.class_nursery import ClassNursery
import numpy as np


@sprinkle_dataset(vocab_pipe=["char_vocab", "word_vocab"])
class sample_dataset(BaseTextClassification, ClassNursery):
    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_instance_length: int,
        word_vocab_store_location: str,
        char_vocab_store_location: str,
        max_char_length: int,
        debug: bool = False,
        debug_dataset_proportion=0.1,
        word_embedding_type: Optional[str] = "random",
        word_embedding_dimension: Optional[int] = "100",
        word_start_token: str = "<SOS>",
        word_end_token: str = "<EOS>",
        word_pad_token: str = "<PAD>",
        word_unk_token: str = "<UNK>",
        word_tokenizer=WordTokenizer(tokenizer="vanilla"),
        word_tokenization_type="vanilla",
        char_embedding_type: Optional[str] = "random",
        char_embedding_dimension: Optional[int] = "25",
        char_start_token: str = " ",
        char_end_token: str = " ",
        char_pad_token: str = " ",
        char_unk_token: str = " ",
        char_tokenizer=CharacterTokenizer(),
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
    ):
        self.classname2idx = self.get_classname2idx()
        self.idx2classname = {
            idx: classname for classname, idx in self.classname2idx.items()
        }
        self.filename = filename
        self.train_size = train_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.dataset_type = dataset_type
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion
        self.max_instance_length = max_instance_length
        self.lines, self.labels = self.get_lines_labels(filename=self.filename)

    def __len__(self):
        return len(self.word_instances)

    def __getitem__(self, idx) -> Dict[str, Any]:
        line = self.lines[idx]
        label = self.labels[idx]

        return self.get_iter_dict(
            lines=line,
            word_vocab=self.word_vocab,
            word_tokenizer=self.word_tokenizer,
            max_word_length=self.max_instance_length,
            char_vocab=self.char_vocab,
            char_tokenizer=self.char_tokenizer,
            max_char_length=self.max_char_length,
            word_add_start_end_token=True,
            labels=label,
        )

    def get_lines_labels(self, filename: str) -> (List[str], List[str]):

        lines = []
        labels = []

        with open(self.filename) as fp:
            for line in fp:
                line = line.strip()
                columns = line.split("\t")
                text, label = columns[0], columns[-1]
                lines.append(text)
                labels.append(label)

        (train_lines, train_labels), (validation_lines, validation_labels), (
            test_lines,
            test_labels,
        ) = self.get_train_valid_test_stratified_split(
            lines, labels, self.classname2idx
        )

        if self.dataset_type == "train":
            lines = train_lines
            labels = train_labels
        elif self.dataset_type == "valid":
            lines = validation_lines
            labels = validation_labels
        elif self.dataset_type == "test":
            lines = test_lines
            labels = test_labels

        if self.debug:
            # randomly sample `self.debug_dataset_proportion`  samples and return
            num_text = len(lines)
            np.random.seed(1729)  # so we can debug deterministically
            random_ints = np.random.randint(
                0, num_text - 1, size=int(self.debug_dataset_proportion * num_text)
            )
            random_ints = list(random_ints)
            sample_texts = []
            sample_labels = []
            for random_int in random_ints:
                sample_texts.append(lines[random_int])
                sample_labels.append(labels[random_int])
            lines = sample_texts
            labels = sample_labels

        return lines, labels

    @classmethod
    def get_classname2idx(cls) -> Dict[str, int]:

        categories = ["class_2", "class_1"]
        classname2idx = [(word, idx) for idx, word in enumerate(categories)]
        classname2idx = dict(classname2idx)
        return classname2idx

    def get_num_classes(self) -> int:
        return len(self.classname2idx.keys())

    def get_class_names_from_indices(self, indices: List) -> List[str]:
        return [self.idx2classname[idx] for idx in indices]

    def emits_keys(cls):
        return {
            "tokens": f"A torch.LongTensor f size max_length where every number represents a number in the vocab",
            "len_tokens": f"Number of tokens in each instance",
            "label": "A torch.LongTensor representing the label corresponding to the instance",
            "instance": f"A string that is padded to max_word_length",
            "raw_instance": f"A sting that is not padded",
            "char_tokens": f"tokens of shape max_word_length * max_char_length representing characters for every word in the instance",
        }

    def print_stats(self):
        pass

    @classmethod
    def get_iter_dict(
        cls,
        lines: Union[List[str], str],
        word_vocab: Vocab,
        word_tokenizer: WordTokenizer,
        max_word_length: int,
        char_vocab: Vocab,
        char_tokenizer: CharacterTokenizer,
        max_char_length: int,
        word_add_start_end_token: bool,
        labels: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(lines, str):
            lines = [lines]

        word_instances = word_tokenizer.tokenize_batch(lines)
        len_instances = [len(instance) for instance in word_instances]
        word_numericalizer = Numericalizer(vocabulary=word_vocab)
        classnames2idx = cls.get_classname2idx()

        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]

                labels = [classnames2idx[label] for label in labels]
                label = torch.LongTensor(labels)

        padded_instances = []
        for word_instance in word_instances:
            padded_instance = pack_to_length(
                tokenized_text=word_instance,
                max_length=max_word_length,
                pad_token=word_vocab.pad_token,
                add_start_end_token=True,
                start_token=word_vocab.start_token,
                end_token=word_vocab.end_token,
            )
            padded_instances.append(padded_instance)

        tokens = word_numericalizer.numericalize_batch_instances(padded_instances)
        tokens = torch.LongTensor(tokens)
        tokens = tokens.squeeze(0)

        instances = []
        for instance in padded_instances:
            instances.append(" ".join(instance))

        raw_instances = []
        for instance in word_instances:
            raw_instances.append(" ".join(instance))

        len_tokens = torch.LongTensor(len_instances)

        # squeeze the dimensions if there are more than one dimension

        if len(instances) == 1:
            instances = instances[0]
            raw_instances = raw_instances[0]

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "instance": instances,
            "raw_instance": raw_instances,
        }

        character_tokens = []
        char_numericalizer = Numericalizer(vocabulary=char_vocab)
        # 1. For every word we get characters in the word
        # 2. Pad the characters to max_char_length
        # 3. Convert them into numbers
        # 4. Add them to character_tokens
        for word in padded_instances:
            char_instance = char_tokenizer.tokenize(word)
            padded_character_instance = pack_to_length(
                tokenized_text=char_instance,
                max_length=max_char_length,
                pad_token=char_vocab.pad_token,
                add_start_end_token=False,
            )
        padded_character_tokens = char_numericalizer.numericalize_instance(
            padded_character_instance
        )
        character_tokens.append(padded_character_tokens)
        character_tokens = torch.LongTensor(character_tokens)
        instance_dict["char_tokens"] = character_tokens

        if labels is not None:
            instance_dict["label"] = label

        return instance_dict
