from abc import ABCMeta, abstractmethod
from typing import Dict, List
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer


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
