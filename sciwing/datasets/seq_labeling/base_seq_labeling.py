from abc import ABCMeta, abstractmethod
from typing import Dict, List
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer


class BaseSeqLabelingDataset(metaclass=ABCMeta):
    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        """ Base Text Classification Dataset to be inherited by all text classification datasets

        Parameters
        ----------
        filename : str
            Path of file where the text classification dataset is stored. Ideally this should have
            an example text and label separated by space. But it is left to the specific dataset to
            handle the different ways in which file could be structured
        tokenizers : Dict[str, BaseTokeizer]
        """

    @abstractmethod
    def get_lines_labels(self) -> (List[Line], List[SeqLabel]):
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
