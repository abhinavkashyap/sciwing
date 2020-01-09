from abc import ABCMeta, abstractmethod
from typing import Dict, List
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class BaseTextClassification(metaclass=ABCMeta):
    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        """ Base Text Classification Dataset to be inherited by all text classification datasets

        Parameters
        ----------
        filename: str
            Full path of the filename where classification dataset is stored
        tokenizers: Dict[str, BaseTokenizer]
            The mapping between namespace and a tokenizer

        """
        pass

    @abstractmethod
    def get_classname2idx(self) -> Dict[str, int]:
        """ Returns the mapping from classname to idx

        Returns
        -------
        Dict[str, int]
            The mapping between class name to idx.

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

    def get_class_names_from_indices(self, indices: List[int]) -> List[str]:
        """ Return a set of class names from indices. Utility method useful for display purposes

        Parameters
        ----------
        indices : List[int]
            List of indices where every index should be between [0, ``num_classes``)

        Returns
        -------
        List[str]
            List of class names for ``indices``
        """
        idx2classnames = {
            idx: classname for classname, idx in self.classnames2idx.items()
        }

        classnames = [idx2classnames[index] for index in indices]

        return classnames

    @abstractmethod
    def get_lines_labels(self) -> (List[Line], List[Label]):
        """ A list of lines from the file and a list of corresponding labels

        This method is to be implemented by a new dataset. The decision on
        the implementation logic is left to the new class. Datasets come in all
        shapes and sizes.

        Parameters
        ---------


        Returns
        -------
        (List[str], List[str])
            Returns a list of text examples and corresponding labels

        """
        pass

    @staticmethod
    def get_train_valid_test_stratified_split(
        lines: List[str],
        labels: List[str],
        classname2idx: Dict[str, int],
        train_size: float = 0.8,
        validation_size: float = 0.5,
        test_size: float = 0.2,
    ) -> ((List[str], List[str]), (List[str], List[str]), (List[str], List[str])):
        len_lines = len(lines)
        len_labels = len(labels)

        assert len_lines == len_labels

        train_test_spliiter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, train_size=train_size, random_state=1729
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
            test_size=validation_size,
            train_size=1 - validation_size,
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
