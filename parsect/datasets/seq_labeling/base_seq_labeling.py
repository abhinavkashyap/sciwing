from abc import ABCMeta, abstractmethod
from typing import Union, Dict, List, Optional
from parsect.tokenizers.word_tokenizer import WordTokenizer
import wasabi
from parsect.tokenizers.character_tokenizer import CharacterTokenizer
from parsect.preprocessing.instance_preprocessing import InstancePreprocessing
from parsect.vocab.vocab import Vocab
import torch


class BaseSeqLabelingDataset(metaclass=ABCMeta):
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
        character_tokenizer=CharacterTokenizer(),
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
            Any of the ``allowed_types`` in vocab.WordEmbLoader is allowed here
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
        character_tokenizer : str
            Any of the ``tokenizer.CharacterTokenizer`` that can be used for character
            tokenization
        """
        self.filename = filename
        self.dataset_type = dataset_type
        self.max_num_words = max_num_words
        self.max_length = max_instance_length
        self.store_location = word_vocab_store_location
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion
        self.embedding_type = word_embedding_type
        self.embedding_dimension = word_embedding_dimension
        self.start_token = word_start_token
        self.end_token = word_end_token
        self.pad_token = word_pad_token
        self.unk_token = word_unk_token
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.word_tokenizer = word_tokenizer
        self.word_tokenization_type = word_tokenization_type
        self.char_tokenizer = character_tokenizer
        self.msg_printer = wasabi.Printer()
        self.allowable_dataset_types = ["train", "valid", "test"]

        self.msg_printer.divider("{0} DATASET".format(self.dataset_type.upper()))

        assert self.dataset_type in self.allowable_dataset_types, (
            "You can Pass one of these "
            "for dataset types: {0}".format(self.allowable_dataset_types)
        )

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

        In sequential labeling, the tagging scheme can be different.
        For example in BIO tagging scheme for NER, the beginning of an
        entity like Person can be B-PER. B-PER counts for one class

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

    def word_tokenize(self, lines: List[str]) -> List[List[str]]:
        """ Tokenize a set of ``lines``.

        Parameters
        ----------
        lines : List[str]
            Word tokenize a set of lines

        Returns
        -------
        List[List[int]]
            Every line is tokenized into a set of words

        """
        instances = list(map(lambda line: self.word_tokenizer.tokenize(line), lines))
        return instances

    def character_tokenize(self, lines: List[str]) -> List[List[str]]:
        """ Character tokenize instances

        Parameters
        ----------
        lines : List[str]
            Character tokenize a set of lines

        Returns
        -------
        List[List[str]]
            Returns the character tokenized sentences

        """
        instances = self.char_tokenizer.tokenize_batch(lines)
        return instances

    @abstractmethod
    def get_lines_labels(self, filename: str) -> (List[str], List[str]):
        """ A list of lines from the file and a list of corresponding labels

        This method is to be implemented by a new dataset. The decision on
        the implementation logic is left to the inheriting class. Datasets come in all
        shapes and sizes.

        For example return ["NUS is a national school", "B-ORG O O O"] for NER


        Returns
        -------
        (List[str], List[str])
            Returns a list of text examples and corresponding labels

        """
        pass

    @abstractmethod
    def get_preloaded_word_embedding(self) -> torch.FloatTensor:
        """ A torch.FloatTensor of 2 dimensions that has embedding values for all the words in vocab

        This is a ``[vocab_len, embedding_dimension]`` matrix, that has embedding values
        for all the words in the vocab

        Returns
        -------
        torch.FloatTensor
            Matrix containing the word representations

        """
        pass

    @abstractmethod
    def get_preloaded_char_embedding(self) -> torch.FloatTensor:
        """ A torch.FloatTensor of 2 dimensions that has embedding values for all the words in vocab

        This is a ``[char_vocab_len, embedding_dimension]`` matrix, that has embedding values
        for all the characters in the vocab

        Returns
        -------
        torch.FloatTensor
            Matrix containing the embeddings for characters

        """
        pass

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
    def get_iter_dict(
        cls,
        line: str,
        word_vocab: Vocab,
        word_tokenizer: WordTokenizer,
        max_word_length: int,
        word_add_start_end_token: bool,
        instance_preprocessor: Optional[InstancePreprocessing] = None,
        char_vocab: Optional[Vocab] = None,
        char_tokenizer: Optional[CharacterTokenizer] = None,
        max_char_length: Optional[int] = None,
        labels: Optional[List[str]] = None,
        need_padding: bool = True,
    ):
        pass
