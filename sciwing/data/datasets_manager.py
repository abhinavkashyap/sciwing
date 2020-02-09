"""
The dataset manager orchestrates different datasets for a problem
It is a container for the train dev and test datasets
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sciwing.vocab.vocab import Vocab
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from typing import Dict, List, Any
from collections import defaultdict
import wasabi
import numpy as np


class DatasetsManager:
    def __init__(
        self,
        train_dataset: Dataset,
        dev_dataset: Dataset = None,
        test_dataset: Dataset = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = None,
        batch_size: int = 32,
        sample_proportion: float = 1.0,
    ):
        """

        Parameters
        ----------
        train_dataset : Dataset
            A pytorch dataset that represents training data
        dev_dataset : Dataset
            A pytorch dataset that represents validation data
        test_dataset : Dataset
            A pytorch dataset that represents test data
        namespace_vocab_options : Dict[str, Dict[str, Any]]
            For every namespace you can give a set of options that will
            be passed down to Vocab.
        namespace_numericalizer_map: Dict[str, Dict[str, Any]]
            For every namespace, you can give a set of options here that will
            be passed down to the Numericalizer Instances
        batch_size: int
            Batch size for loading the datasets
        sample_proportion: int
            The sample proportion is used to provide a smaller datasets
            from the original dataset. This helps you debug your models easily.
            This has to be between 0 and 1. If it is 1 we use all the training
            data for training
        """
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.sample_proportion = sample_proportion
        self.label_namespaces: List[str] = None  # Holds the label namespaces
        self.msg_printer = wasabi.Printer()

        assert 0.0 < self.sample_proportion <= 1.0

        if namespace_vocab_options is None:
            self.namespace_vocab_options = {}
        else:
            self.namespace_vocab_options = namespace_vocab_options

        self.batch_size = batch_size

        self.namespace_to_numericalizer: Dict[
            str, BaseNumericalizer
        ] = namespace_numericalizer_map

        # get train lines and labels
        self.train_lines, self.train_labels = self.get_train_lines_labels()

        # Build vocab using the datasets passed
        self.namespace_to_vocab = self.build_vocab()

        # sets the vocab for the appropriate numericalizers
        self.namespace_to_numericalizer = self.build_numericalizers()

        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, collate_fn=list
        )

        self.dev_loader = DataLoader(
            dataset=self.dev_dataset, batch_size=self.batch_size, collate_fn=list
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, collate_fn=list
        )

        self.train_loader = iter(self.train_loader)
        self.dev_loader = iter(self.dev_loader)
        self.test_loader = iter(self.test_loader)
        self.namespaces = list(self.namespace_to_vocab.keys())
        self.num_labels = {}
        for namespace in self.label_namespaces:
            self.num_labels[namespace] = self.namespace_to_vocab[
                namespace
            ].get_vocab_len()

    def build_vocab(self) -> Dict[str, Vocab]:
        """ Returns a vocab for each of the namespace
        The namespace identifies the kind of tokens
        Some tokens correspond to words
        Some tokens may correspond to characters.
        Some tokens may correspond to Bert style tokens

        Returns
        -------
        Dict[str, Vocab]
            A vocab corresponding to each of the

        """
        lines = self.train_lines
        labels = self.train_labels
        namespace_to_instances: Dict[str, List[List[str]]] = defaultdict(list)
        for line in lines:
            namespace_tokens = line.tokens
            for namespace, tokens in namespace_tokens.items():
                tokens = [tok.text for tok in tokens]
                namespace_to_instances[namespace].append(tokens)
        for label in labels:
            namespace_tokens = label.tokens
            for namespace, tokens in namespace_tokens.items():
                tokens = [tok.text for tok in tokens]
                namespace_to_instances[namespace].append(tokens)

        self.label_namespaces = list(labels[0].tokens.keys())

        namespace_to_vocab: Dict[str, Vocab] = {}

        # This always builds a vocab from instances
        for namespace, instances in namespace_to_instances.items():
            namespace_to_vocab[namespace] = Vocab(
                instances=instances, **self.namespace_vocab_options.get(namespace, {})
            )
            namespace_to_vocab[namespace].build_vocab()
        return namespace_to_vocab

    def get_train_lines_labels(self) -> (List[Line], List[Label]):
        """ Returns training lines and labels. Samples the training lines
        and labels according to sample proportion and returns it

        Returns
        -------
        (List[Line], List[Label])

        """
        lines, labels = self.train_dataset.get_lines_labels()
        if self.sample_proportion != 1.0:
            len_lines = len(lines)
            lines_ = []
            labels_ = []
            np.random.seed(1729)
            sample_size = np.ceil(self.sample_proportion * len_lines)
            sample_size = int(sample_size)
            random_indices = np.random.randint(low=0, high=len_lines, size=sample_size)
            for random_idx in random_indices:
                line = lines[random_idx]
                label = labels[random_idx]
                lines_.append(line)
                labels_.append(label)
            return lines_, labels_
        else:
            return lines, labels

    def print_stats(self):
        """ Print different stats with respect to the train, dev and test datasets

        Returns
        -------
        None

        """
        len_train_dataset = len(self.train_dataset)
        len_dev_dataset = len(self.dev_dataset)
        len_test_dataset = len(self.test_dataset)

        self.msg_printer.info(f"Num of training examples: {len_train_dataset}")
        self.msg_printer.info(f"Num of dev examples {len_dev_dataset}")
        self.msg_printer.info(f"Num of test examples {len_test_dataset}")

        # print namespace to vocab stats
        for namespace in self.namespaces:
            vocab = self.namespace_to_vocab[namespace]
            self.msg_printer.divider(text=f"Namespace {namespace}")
            vocab.print_stats()

    def build_numericalizers(self):
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = {}
        for namespace, numericalizer in self.namespace_to_numericalizer.items():
            numericalizer.vocabulary = self.namespace_to_vocab[namespace]
            namespace_numericalizer_map[namespace] = numericalizer

        return namespace_numericalizer_map

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def dev_dataset(self):
        return self._dev_dataset

    @dev_dataset.setter
    def dev_dataset(self, value):
        self._dev_dataset = value

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def num_labels(self):
        return self._num_labels

    @num_labels.setter
    def num_labels(self, value):
        self._num_labels = value

    @property
    def namespace_to_vocab(self):
        return self._namespace_to_vocab

    @namespace_to_vocab.setter
    def namespace_to_vocab(self, value):
        self._namespace_to_vocab = value

    @property
    def namespaces(self):
        return self._namespaces

    @namespaces.setter
    def namespaces(self, value):
        self._namespaces = value

    def get_idx_label_mapping(self):
        pass

    @property
    def label_namespaces(self):
        return self._label_namespaces

    @label_namespaces.setter
    def label_namespaces(self, value):
        self._label_namespaces = value
