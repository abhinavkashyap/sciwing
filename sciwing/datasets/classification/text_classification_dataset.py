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
        lines = []
        labels = []

        with open(self.filename) as fp:
            for line in fp:
                line, label = line.split("###")
                line = line.strip()
                label = label.strip()
                line_instance = Line(text=line, tokenizers=self.tokenizers)
                label_instance = Label(label_str=label)
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
        tokenizers: Dict[str, BaseTokenizer] = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        batch_size: int = 1,
    ):
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.batch_size = batch_size

        if tokenizers is None:
            self.tokenizers = {"tokens": WordTokenizer()}

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
            namespace_vocab_options=namespace_vocab_options,
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
        namespaces = list(lines[0].tokenizers.keys())

        namespace_to_numericalized = defaultdict(list)

        namespace_to_numericalizer = self.namespace_to_numericalizer
        namespace_to_vocab = self.namespace_to_vocab

        for line in lines:
            for namespace in namespaces:
                numericalizer = namespace_to_numericalizer[namespace]
                numericalized = numericalizer.numericalize_instance(
                    line.tokens[namespace]
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
