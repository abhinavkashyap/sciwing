from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List
from parsect.tokenizers.word_tokenizer import WordTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from torch.utils.data import Dataset


class BaseTextClassification(Dataset, metaclass=ABCMeta):
    @abstractmethod
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
        word_tokenizer=WordTokenizer(),
        word_tokenization_type="vanilla",
    ):
        """ Base Text Classification Dataset to be inherited by all text classification datasets

        Parameters
        ----------
        filename : str
            Path of file where the text classification dataset is stored. Ideally this should have
            an example text and label separated by space. But it is left to the specific dataset to
            handle the different ways in which file could be structured
        dataset_type : str
            One of ``[train, valid, test]``
        max_num_words : int
            The top ``max_num_words`` will be considered for building vocab
        max_instance_length : int
            Every instance in the dataset will be padded to or curtailed to ``max_length`` number of
            tokens
        word_vocab_store_location : str
            Vocabulary once built will be stored in this location
            If the vocabulary already exists then it will be loaded from the filepath
        debug : bool
            Useful to build a small dataset for debugging purposes. If ``True``, then a smaller
            random version of the dataset should be returned. If ``True`` then
            ``debug_dataset_proportion`` will be the proportion of the dataset that will be returned
        debug_dataset_proportion : int
            Percent of dataset that will be returned for debug purposes. Should be between 0 and 1
        word_embedding_type : str
            The kind of word embedding that will be associated with the words in the database
            Any of the ``allowed_types`` in vocab.EmbeddingLoader is allowed here
        word_embedding_dimension : int
            Dimension of word embedding
        word_start_token : str
            Start token appended at the beginning of every instance
        word_end_token : str
            End token appended at the end of every instance
        word_pad_token : str
            Pad token to be used for padding
        word_unk_token : str
            All OOV words (if they are less frequent than ``max_words`` or word is in
            test but not in train) will be mapped to ``unk_token``
        train_size : str
            Percentage of the instances to be used for training
        test_size : str
            Remaining percentage that will be used for testing
        validation_size : str
            Percentage of test data that will be used for validation
        word_tokenizer : WordTokenizer
            Word Tokenizer to be used for the dataset. You can reference
            ``tokenizers.WordTokenizer`` for more information
        word_tokenization_type : str
            The type of word tokenization that the word tokenizer represents
        """
        Dataset.__init__()

    @classmethod
    @abstractmethod
    def get_classname2idx(cls) -> Dict[str, int]:
        """ Mapping between classnames and index

        Returns
        -------
        Dict[str, int]
            A mapping between class names and idx
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """ Return the number of classes in the dataset

        Returns
        -------
        int
            Number of classes in the dataset
        """
        pass

    @abstractmethod
    def get_class_names_from_indices(self, indices: List[int]) -> List[str]:
        """ Return a set of class names from indices. Utility method useful for display purposes

        Parameters
        ----------
        indices : List[int]
            List of indices where every index should be between [0, ``num_classes``)

        Returns
        -------
        List[str]
            List of class names for indices
        """
        pass

    @abstractmethod
    def print_stats(self):
        pass

    @abstractmethod
    def get_lines_labels(self, filename: str) -> (List[str], List[str]):
        """ A list of lines from the file and a list of corresponding labels

        This method is to be implemented by a new dataset. The decision on
        the implementation logic is left to the new class. Datasets come in all
        shapes and sizes.

        Parameters
        ---------
        filename
            The filename where the dataset is stored

        Returns
        -------
        (List[str], List[str])
            Returns a list of text examples and corresponding labels

        """
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

    @classmethod
    @abstractmethod
    def emits_keys(cls) -> Dict[str, str]:
        """ Specify the keys that will be emitted in the instance dict fo the dataset

        The ``instance_dict`` is a dictionary of string to tensors. The ``instance_dict`` can
        contain various keys depending on the dataset that is being used and the model that
        is built using the dataset. The function should provides means to inspect the different
        keys emitted by the classification dataset and the description of what they mean

        Returns
        -------
        Dict[str, str]
            A dictionary representing different keys emitted and their corresponding human
            readable description
        """
        pass

    @classmethod
    def get_iter_dict(cls, *args, **kwargs):
        pass
