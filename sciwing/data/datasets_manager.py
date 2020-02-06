"""
The dataset manager orchestrates different datasets for a problem
It is a container for the train dev and test datasets
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sciwing.vocab.vocab import Vocab
from sciwing.numericalizers.numericalizer import Numericalizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from typing import Dict, List, Any, Optional
from collections import defaultdict


class DatasetsManager:
    def __init__(
        self,
        train_dataset: Dataset,
        dev_dataset: Dataset = None,
        test_dataset: Dataset = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = None,
        batch_size: int = 32,
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
        """
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        if namespace_vocab_options is None:
            self.namespace_vocab_options = {}
        else:
            self.namespace_vocab_options = namespace_vocab_options

        self.batch_size = batch_size

        self.namespace_to_numericalizer: Dict[
            str, BaseNumericalizer
        ] = namespace_numericalizer_map

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
        lines, labels = self.train_dataset.get_lines_labels()
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

        namespace_to_vocab: Dict[str, Vocab] = {}

        # This always builds a vocab from instances
        for namespace, instances in namespace_to_instances.items():
            namespace_to_vocab[namespace] = Vocab(
                instances=instances, **self.namespace_vocab_options.get(namespace, {})
            )
            namespace_to_vocab[namespace].build_vocab()
        return namespace_to_vocab

    def print_stats(self):
        """ Print different stats with respect to the train, dev and test datasets

        Returns
        -------
        None

        """
        pass

    def build_numericalizers(self):
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = {}
        for namespace, numericalizer in self.namespace_to_numericalizer.items():
            numericalizer.vocabulary = self.namespace_to_vocab[namespace]
            namespace_numericalizer_map[namespace] = numericalizer

        return namespace_numericalizer_map

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
