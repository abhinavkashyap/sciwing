from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List
from parsect.tokenizers.word_tokenizer import WordTokenizer
import wasabi
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class TextClassificationDataset(metaclass=ABCMeta):
    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_length: int,
        vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        embedding_type: Union[str, None] = None,
        embedding_dimension: Union[int, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        tokenizer=WordTokenizer(),
        tokenization_type="vanilla",
    ):
        """
               :param dataset_type: type: str
               One of ['train', 'valid', 'test']
               :param max_num_words: type: int
               The top frequent `max_num_words` to consider
               :param max_length: type: int
               The maximum length after numericalization
               :param vocab_store_location: type: str
               The vocab store location to store vocabulary
               This should be a json filename
               :param debug: type: bool
               If debug is true, then we randomly sample
               10% of the dataset and work with it. This is useful
               for faster automated tests and looking at random
               examples
               :param debug_dataset_proportion: type: float
               Send a number (0.0, 1.0) and a random proportion of the dataset
               will be used for debug purposes
               :param embedding_type: type: str
               Pre-loaded embedding type to load.
               :param start_token: type: str
               The start token is the token appended to the beginning of the list of tokens
               :param end_token: type: str
               The end token is the token appended to the end of the list of tokens
               :param pad_token: type: str
               The pad token is used when the length of the input is less than maximum length
               :param unk_token: type: str
               unk is the token that is used when the word is OOV
               :param train_size: float
               The proportion of the dataset that is used for training
               :param test_size: float
               The proportion of the dataset that is used for testing
               :param validation_size: float
               The proportion of the test dataset that is used for validation
               :param tokenizer
               The tokenizer that will be used to tokenize text
               :param tokenization_type: str
               Allowed type (vanilla, bert)
               Two types of tokenization are allowed. Either vanilla tokenization that is based on spacy.
               The default is WordTokenizer()
               If bert, then bert tokenization is performed and additional fields will be included in the output

               """
        self.filename = filename
        self.dataset_type = dataset_type
        self.max_num_words = max_num_words
        self.max_length = max_length
        self.store_location = vocab_store_location
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion
        self.embedding_type = embedding_type
        self.embedding_dimension = embedding_dimension
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.tokenizer = tokenizer
        self.tokenization_type = tokenization_type
        self.msg_printer = wasabi.Printer()
        self.allowable_dataset_types = ["train", "valid", "test"]

        self.msg_printer.divider("{0} DATASET".format(self.dataset_type.upper()))

        assert self.dataset_type in self.allowable_dataset_types, (
            "You can Pass one of these "
            "for dataset types: {0}".format(self.allowable_dataset_types)
        )

    @abstractmethod
    def get_label_mapping(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abstractmethod
    def get_class_names_from_indices(self, indices: List[int]):
        pass

    @abstractmethod
    def get_disp_sentence_from_indices(self, indices: List[int]):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    def tokenize(self, lines: List[str]) -> List[List[str]]:
        """
        :param lines: type: List[str]
        These are text spans that will be tokenized
        :return: instances type: List[List[str]]
        """
        instances = list(map(lambda line: self.tokenizer.tokenize(line), lines))
        return instances

    @abstractmethod
    def get_lines_labels(self):
        pass

    @abstractmethod
    def get_preloaded_embedding(self):
        pass

    def get_train_valid_test_stratified_split(
        self, lines: List[str], labels: List[str], classname2idx: Dict[str, int]
    ) -> ((List[str], List[str]), (List[str], List[str]), (List[str], List[str])):
        len_lines = len(lines)
        len_labels = len(labels)

        assert len_lines == len_labels

        train_test_spliiter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=1729,
        )

        features = np.random.rand(len_lines)
        labels_idx_array = np.array([classname2idx[label] for label in labels])

        splits = list(train_test_spliiter.split(features, labels_idx_array))
        train_indices, test_valid_indices = splits[0]

        train_lines = [lines[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]

        test_valid_lines = [lines[idx] for idx in test_valid_indices]
        test_valid_labels = [labels[idx] for idx in test_valid_indices]

        validation_test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            train_size=1 - self.validation_size,
            random_state=1729,
        )

        len_test_valid_lines = len(test_valid_lines)
        len_test_valid_labels = len(test_valid_labels)

        assert len_test_valid_labels == len_test_valid_lines

        test_valid_features = np.random.rand(len_test_valid_lines)
        test_valid_labels_idx_array = np.array(
            [classname2idx[label] for label in test_valid_labels]
        )

        test_valid_splits = list(
            validation_test_splitter.split(
                test_valid_features, test_valid_labels_idx_array
            )
        )
        test_indices, validation_indices = test_valid_splits[0]

        test_lines = [test_valid_lines[idx] for idx in test_indices]
        test_labels = [test_valid_labels[idx] for idx in test_indices]

        validation_lines = [test_valid_lines[idx] for idx in validation_indices]
        validation_labels = [test_valid_labels[idx] for idx in validation_indices]

        return (
            (train_lines, train_labels),
            (validation_lines, validation_labels),
            (test_lines, test_labels),
        )
