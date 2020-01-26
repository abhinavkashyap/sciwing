from typing import Dict, List, Any, Tuple, Optional
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from torch.utils.data import Dataset
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from sciwing.data.datasets_manager import DatasetsManager
from collections import defaultdict
import operator
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizer.numericalizer import Numericalizer
from sciwing.numericalizer.transformer_numericalizer import NumericalizerForTransformer


class TextClassificationDataset(BaseTextClassification, Dataset):
    """ This represents a dataset that is of the form
    line1###label1
    line2###label2
    line3###label3
    .
    .
    .
    """

    def get_num_classes(self) -> int:
        return len(self.classnames2idx)

    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.classnames2idx: Dict[str, int] = self.get_classname2idx()
        self.lines, self.labels = self.get_lines_labels()

    def get_classname2idx(self) -> Dict[str, int]:
        classnames = set()
        with open(self.filename) as fp:
            for line in fp:
                try:
                    line, label = line.split("###")
                    label = label.strip()
                    classnames.add(label)
                except ValueError:
                    print(
                        f"Check the format of the file {self.filename}. Every line should be of the "
                        f"format line###label"
                    )

        classnames = list(classnames)
        classname2idx = {classname: idx for idx, classname in enumerate(classnames)}
        return classname2idx

    def get_lines_labels(self) -> (List[Line], List[Label]):
        lines: List[Line] = []
        labels: List[Label] = []

        with open(self.filename) as fp:
            for line in fp:
                line, label = line.split("###")
                line = line.strip()
                label = label.strip()
                line_instance = Line(text=line, tokenizers=self.tokenizers)
                label_instance = Label(text=label)
                lines.append(line_instance)
                labels.append(label_instance)

        return lines, labels

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (Line, Label):
        line, label = self.lines[idx], self.labels[idx]
        return line, label


class TextClassificationDatasetManager(DatasetsManager):
    def __init__(
        self,
        train_filename: str,
        dev_filename: str = None,
        test_filename: str = None,
        batch_size: int = 1,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
    ):
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.batch_size = batch_size

        # For the classification task we will device the tokenizers
        # and the namespace vocab options
        # This provides a clean API for the users to perform classification
        # to pass the files in a clean manner
        # TODO: This will be pretty much the same for all the datasets. Move this to the DatasetsManager

        # we will also give appropriate names to namespaces
        # The whole things takes control away from the user
        # but, we choose to restrict the user to use only these options for text
        # classification in the scientific document processing domain

        self.word_tokenizer = WordTokenizer()
        self.char_tokenizer = CharacterTokenizer()
        self.bert_base_uncased_tokenizer = TokenizerForBert(
            bert_type="bert-base-uncased"
        )
        self.bert_base_cased_tokenizer = TokenizerForBert(bert_type="bert-base-cased")
        self.scibert_base_uncased_tokenizer = TokenizerForBert(
            bert_type="scibert-base-uncased"
        )
        self.scibert_base_cased_tokenizer = TokenizerForBert(
            bert_type="scibert-base-cased"
        )

        self.word_numericalizer = Numericalizer()
        self.char_tokens_numericalizer = Numericalizer()
        self.bert_base_uncased_numericalizer = NumericalizerForTransformer(
            tokenizer=self.bert_base_uncased_tokenizer
        )
        self.bert_base_cased_numericalizer = NumericalizerForTransformer(
            tokenizer=self.bert_base_cased_tokenizer
        )
        self.scibert_base_uncased_numericalizer = NumericalizerForTransformer(
            tokenizer=self.scibert_base_uncased_tokenizer
        )
        self.scibert_base_cased_numericalizer = NumericalizerForTransformer(
            tokenizer=self.scibert_base_cased_tokenizer
        )
        self.label_numericalizer = Numericalizer()

        self.tokenizers = {
            "tokens": self.word_tokenizer,
            "char_tokens": self.char_tokenizer,
            "bert_base_uncased_tokens": self.bert_base_uncased_tokenizer,
            "bert_base_cased_tokens": self.bert_base_cased_tokenizer,
            "scibert_base_cased_tokens": self.scibert_base_cased_tokenizer,
            "scibert_base_uncased_tokens": self.scibert_base_cased_tokenizer,
        }

        self.namespace_vocab_options = namespace_vocab_options or {
            "tokens": {"max_instance_length": 100},
            "char_tokens": {
                "max_instance_length": 15,
                "unk_token": "",
                "pad_token": "",
                "start_token": "",
                "end_token": "",
            },
        }

        self.namespace_numericalizer_map = {
            "tokens": self.word_numericalizer,
            "char_tokens": self.char_tokens_numericalizer,
            "bert_base_uncased_tokens": self.bert_base_uncased_numericalizer,
            "bert_base_cased_tokens": self.bert_base_cased_numericalizer,
            "scibert_base_cased_tokens": self.scibert_base_uncased_numericalizer,
            "scibert_base_uncased_tokens": self.scibert_base_cased_numericalizer,
            "label": self.label_numericalizer,
        }

        self.train_dataset = TextClassificationDataset(
            filename=self.train_filename, tokenizers=self.tokenizers
        )
        self.dev_dataset = TextClassificationDataset(
            filename=self.dev_filename, tokenizers=self.tokenizers
        )
        self.test_dataset = TextClassificationDataset(
            filename=self.test_filename, tokenizers=self.tokenizers
        )

        super(TextClassificationDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )

    def get_iter_dict(self, lines: List[Line], labels: Optional[List[str]] = None):
        """ Returns the iterdict for the set of lines and labels

        Parameters
        ----------
        lines : List[Line]
            A list of lines to get the iterdict
        labels : List[Label]
            A list of labels corresponding to the
        Returns
        -------
        Dict[str, Any]

        """
        if isinstance(lines, Line):
            lines = [lines]

        if isinstance(labels, Label):
            labels = [labels]

        # get all namespaces for the line
        line_namespaces = list(lines[0].tokenizers.keys())

        namespace_to_numericalized = defaultdict(list)

        namespace_to_numericalizer = self.namespace_to_numericalizer
        namespace_to_vocab = self.namespace_to_vocab

        for line in lines:
            for namespace in line_namespaces:
                numericalizer = namespace_to_numericalizer[namespace]
                line_tokens = line.tokens[namespace]
                line_tokens = [tok.text for tok in line_tokens]
                numericalized = numericalizer.numericalize_instance(line_tokens)
                numericalized = numericalizer.pad_instance(
                    numericalized,
                    max_length=namespace_to_vocab[namespace].max_instance_length,
                )
                namespace_to_numericalized[namespace].append(numericalized)

        # The label namespace is usually just a single one
        label_namespaces = list(labels[0].tokens.keys())

        for label in labels:
            for namespace in label_namespaces:
                numericalizer = namespace_to_numericalizer[namespace]
                numericalized = numericalizer.numericalize_instance(
                    label.tokens[namespace]
                )
                numericalized = numericalizer.pad_instance(
                    numericalized,
                    max_length=namespace_to_vocab[namespace].max_instance_length,
                )
                namespace_to_numericalized[namespace].append(numericalized)

        return namespace_to_numericalized

    def _get_iter_dict(self, for_dataset="train") -> Dict[str, Any]:
        line_labels: List[Tuple[Line, Label]] = []

        if for_dataset == "train":
            line_labels = next(self.train_loader)
        elif for_dataset == "dev":
            line_labels = next(self.dev_loader)
        elif for_dataset == "test":
            line_labels = next(self.test_loader)

        # The classification dataset returns a list of (Line, Label)
        # The batch of lines and labels are then obtained and
        # the iter dict is obtained
        batch_lines = map(operator.itemgetter(0), line_labels)
        batch_labels = map(operator.itemgetter(1), line_labels)
        batch_lines = list(batch_lines)
        batch_labels = list(batch_labels)

        return self.get_iter_dict(lines=batch_lines, labels=batch_labels)
