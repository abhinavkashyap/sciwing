import torch
from typing import List, Dict, Union, Any, Optional
import sciwing.constants as constants
from sciwing.utils.common import convert_sectlabel_to_json
from sciwing.utils.common import pack_to_length
from wasabi import Printer
import numpy as np
from sciwing.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from sciwing.datasets.sprinkle_dataset import sprinkle_dataset
from sciwing.utils.class_nursery import ClassNursery

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


@sprinkle_dataset()
class SectLabelDataset(BaseTextClassification, ClassNursery):
    """ SectLabel Dataset - A logical section classification dataset from WING-NUS

        Parameters
        ----------
        filename : str
            Name of the file where the SectLabel dataset is stored
        dataset_type : str
            Either of `[train, valid, test]` that this dataset represents
        max_num_words : int
            Maximum number of words to be included in the vocab. The top most frequent
            ``max_num_words`` will be included in the vocab. Everything else will be mapped to
            ``word_unk`` tag.
        max_instance_length : int
            The maximum length for every instance
        word_vocab_store_location : str
            The path where the word vocabulary will be stored
        debug : bool
            Is this dataset a debug dataset where a small portion will be used for testing purposes.
        debug_dataset_proportion : float
            The proportion of the dataset that would be used as debug dataset
        word_embedding_type : str
            The embedding type is any of those that are accepted in ``vocab.embedding_loader``
        word_embedding_dimension : int
            Word embedding size. This might depend on the ``embedding_type`` that is used.
        word_start_token : str
            Every instance will be prepended with a ``word_start_token``
        word_end_token : str
            Every instance will be appended with a ``word_end_token``
        word_pad_token : str
            Token used for padding instances
        word_unk_token : str
            If word is not found in the training vocab, then the word
            is replaced with ``word_unk_token``
        train_size : int
            The portion of the dataset that will be used for training
        test_size : int
            The portion of the dataset that will be used for testing
        validation_size : int
            The portion of the dataset that will be used for validation
        word_tokenization_type : int
            The kind of word tokenization. ``tokenizers.word_tokenizer`` has more information
        add_start_end_token : bool
            Whether starting and ending token should be added to the instance
    """

    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_instance_length: int,
        word_vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[int, None] = None,
        word_start_token: str = "<SOS>",
        word_end_token: str = "<EOS>",
        word_pad_token: str = "<PAD>",
        word_unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
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

        self.msg_printer = Printer()

    def __len__(self) -> int:
        return len(self.word_instances)

    def __getitem__(self, idx) -> Dict[str, Any]:
        line = self.lines[idx]
        label = self.labels[idx]

        return self.get_iter_dict(lines=line, labels=label)

    def get_lines_labels(self, filename: str) -> (List[str], List[str]):
        parsect_json = convert_sectlabel_to_json(filename)
        texts = []
        labels = []
        parsect_json = parsect_json["parse_sect"]

        for line_json in parsect_json:
            text = line_json["text"]
            label = line_json["label"]

            texts.append(text)
            labels.append(label)

        (train_lines, train_labels), (validation_lines, validation_labels), (
            test_lines,
            test_labels,
        ) = self.get_train_valid_test_stratified_split(
            texts, labels, self.classname2idx
        )

        if self.dataset_type == "train":
            texts = train_lines
            labels = train_labels
        elif self.dataset_type == "valid":
            texts = validation_lines
            labels = validation_labels
        elif self.dataset_type == "test":
            texts = test_lines
            labels = test_labels

        if self.debug:
            # randomly sample `self.debug_dataset_proportion`  samples and return
            num_text = len(texts)
            np.random.seed(1729)  # so we can debug deterministically
            random_ints = np.random.randint(
                0, num_text - 1, size=int(self.debug_dataset_proportion * num_text)
            )
            random_ints = list(random_ints)
            sample_texts = []
            sample_labels = []
            for random_int in random_ints:
                sample_texts.append(texts[random_int])
                sample_labels.append(labels[random_int])
            texts = sample_texts
            labels = sample_labels

        return texts, labels

    @classmethod
    def get_classname2idx(cls) -> Dict[str, int]:
        categories = [
            "address",
            "affiliation",
            "author",
            "bodyText",
            "category",
            "construct",
            "copyright",
            "email",
            "equation",
            "figure",
            "figureCaption",
            "footnote",
            "keyword",
            "listItem",
            "note",
            "page",
            "reference",
            "sectionHeader",
            "subsectionHeader",
            "subsubsectionHeader",
            "tableCaption",
            "table",
            "title",
        ]
        categories = [(word, idx) for idx, word in enumerate(categories)]
        categories = dict(categories)
        return categories

    def get_num_classes(self) -> int:
        return len(self.classname2idx.keys())

    def get_class_names_from_indices(self, indices: List) -> List[str]:
        """ Given a list of indices maps back to classnames
        Mostly used for inference and other higher level applications

        Parameters
        ----------
        indices : List[int]
            A list of indices where every index is in ``[0, num_classes)``

        Returns
        -------

        """
        return [self.idx2classname[idx] for idx in indices]

    def print_stats(self):
        num_instances = self.num_instances
        formatted = self.label_stats_table
        self.msg_printer.divider("Stats for {0} dataset".format(self.dataset_type))
        print(formatted)
        self.msg_printer.info(
            f"Number of instances in {self.dataset_type} dataset - {num_instances}"
        )

    def emits_keys(cls):
        return {
            "tokens": f"A torch.LongTensor of size `max_length`. "
            f"Example [0, 0, 1, 100] where every number represents an index in the vocab",
            "len_tokens": f"A torch.LongTensor. "
            f"Example [2] representing the number of tokens without padding",
            "label": f"A torch.LongTensor representing the label corresponding to the "
            f"instance. Example [2] representing class 2",
            "instance": f"A string that is padded to ``max_length``.",
            "raw_instance": f"A string that is not padded",
        }

    def get_iter_dict(
        self,
        lines: Union[List[str], str],
        labels: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(lines, str):
            lines = [lines]

        word_instances = self.word_tokenizer.tokenize_batch(lines)
        len_instances = [len(instance) for instance in word_instances]
        classnames2idx = SectLabelDataset.get_classname2idx()

        padded_instances = []
        for word_instance in word_instances:
            padded_instance = pack_to_length(
                tokenized_text=word_instance,
                max_length=self.max_instance_length,
                pad_token=self.word_vocab.pad_token,
                add_start_end_token=True,
                start_token=self.word_vocab.start_token,
                end_token=self.word_vocab.end_token,
            )
            padded_instances.append(padded_instance)

        tokens = self.word_numericalizer.numericalize_batch_instances(padded_instances)
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

        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]

            labels = [classnames2idx[label] for label in labels]
            label = torch.LongTensor(labels)
            instance_dict["label"] = label

        return instance_dict
