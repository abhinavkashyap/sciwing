from parsect.datasets.sprinkle_dataset import sprinkle_dataset
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from typing import Dict, Any, Optional, Union, List
from parsect.vocab.vocab import Vocab
from torch.utils.data import Dataset
import torch
from parsect.tokenizers.word_tokenizer import WordTokenizer
import numpy as np


@sprinkle_dataset()
class sample_clf_dataset(Dataset, BaseTextClassification):
    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_instance_length: int,
        debug: bool = False,
        debug_dataset_proportion=0.1,
        word_embedding_type: Optional[str] = "random",
        word_embedding_dimension: Optional[int] = "100",
        word_start_token: str = "<SOS>",
        word_end_token: str = "<EOS>",
        word_pad_token: str = "<PAD>",
        word_unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenizer=WordTokenizer(),
        word_tokenization_type="vanilla",
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

    def get_disp_sentence_from_indices(self, indices: List) -> str:
        pass

    def get_preloaded_word_embedding(self) -> torch.FloatTensor:
        pass

    def emits_keys(cls):
        pass

    @classmethod
    def get_iter_dict(
        cls,
        lines: Union[List[str], str],
        word_vocab: Vocab,
        word_tokenizer: WordTokenizer,
        max_word_length: int,
        word_add_start_end_token: bool,
        labels: Optional[Union[str, List[str]]] = None,
    ):
        pass
