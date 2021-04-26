from sciwing.datasets.summarization.base_text_summarization import (
    BaseAbstractiveTextSummarization,
)
from torch.utils.data import Dataset
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from sciwing.numericalizers.numericalizer import Numericalizer
from typing import Dict, List, Any
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.vocab.vocab import Vocab
from collections import defaultdict
from sciwing.data.token import Token


class AbstractiveSummarizationDataset(BaseAbstractiveTextSummarization, Dataset):
    """ This represents a dataset that is of the form
        doc1word1 doc1word2 doc1word3###ref1word1 ref1word2
        doc2word1 doc2word2 doc2word3###ref2word1 ref2word2
        doc3word1 doc3word2 doc3word3###ref3word1 ref3word2
        .
        .
        .
    """

    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(
        self, start_token: str = "<SOS>", end_token: str = "<EOS>"
    ) -> (List[Line], List[Line]):
        lines: List[Line] = []
        labels: List[Line] = []

        with open(self.filename) as fp:
            for line in fp:
                line, label = line.split("###")
                line = line.strip()
                label = label.strip()
                line_instance = Line(text=line, tokenizers=self.tokenizers)
                label_instance = Line(text=label, tokenizers=self.tokenizers)
                for namespace, tokenizer in self.tokenizers.items():
                    line_instance.tokens[namespace].insert(0, Token(start_token))
                    line_instance.tokens[namespace].append(Token(end_token))
                    label_instance.tokens[namespace].insert(0, Token(start_token))
                    label_instance.tokens[namespace].append(Token(end_token))
                lines.append(line_instance)
                labels.append(label_instance)

        return lines, labels

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (Line, Line):
        line, label = self.lines[idx], self.labels[idx]
        return line, label


class AbstractiveSummarizationDatasetManager(DatasetsManager):
    def __init__(
        self,
        train_filename: str,
        dev_filename: str,
        test_filename: str,
        tokenizers: Dict[str, BaseTokenizer] = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = None,
        batch_size: int = 10,
    ):

        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.tokenizers = tokenizers or {
            "tokens": WordTokenizer(tokenizer="vanilla"),
            # "char_tokens": CharacterTokenizer(),
        }
        self.namespace_vocab_options = namespace_vocab_options or {
            "char_tokens": {
                "start_token": " ",
                "end_token": " ",
                "pad_token": " ",
                "unk_token": " ",
            }
        }
        self.namespace_numericalizer_map = namespace_numericalizer_map or {
            "tokens": Numericalizer(),
            # "char_tokens": Numericalizer(),
            "shared_tokens": Numericalizer(),
        }

        self.batch_size = batch_size

        self.train_dataset = AbstractiveSummarizationDataset(
            filename=self.train_filename, tokenizers=self.tokenizers
        )

        self.dev_dataset = AbstractiveSummarizationDataset(
            filename=self.dev_filename, tokenizers=self.tokenizers
        )

        self.test_dataset = AbstractiveSummarizationDataset(
            filename=self.test_filename, tokenizers=self.tokenizers
        )

        super(AbstractiveSummarizationDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )

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
        lines = self.train_dataset.lines
        labels = self.train_dataset.labels

        namespace_to_instances: Dict[str, List[List[str]]] = defaultdict(list)
        shared_namespace: str = "shared_tokens"
        for line in lines:
            namespace_tokens = line.tokens
            for namespace, tokens in namespace_tokens.items():
                namespace_to_instances[namespace].append(tokens)
        for label in labels:
            namespace_tokens = label.tokens
            for namespace, tokens in namespace_tokens.items():
                namespace_to_instances[namespace].append(tokens)
        namespace_list = list(namespace_to_instances.keys()).copy()
        for namespace in namespace_list:
            namespace_to_instances[shared_namespace].extend(
                namespace_to_instances[namespace]
            )

        self.label_namespaces = list(labels[0].tokens.keys())

        namespace_to_vocab: Dict[str, Vocab] = {}

        # This always builds a vocab from instances
        for namespace, instances in namespace_to_instances.items():
            namespace_to_vocab[namespace] = Vocab(
                instances=instances, **self.namespace_vocab_options.get(namespace, {})
            )
            namespace_to_vocab[namespace].build_vocab()
        return namespace_to_vocab
