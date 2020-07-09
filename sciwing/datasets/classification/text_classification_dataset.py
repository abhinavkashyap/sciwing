from typing import Dict, List, Any
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from torch.utils.data import Dataset
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from sciwing.numericalizers.numericalizer import Numericalizer
from sciwing.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.utils.class_nursery import ClassNursery


class TextClassificationDataset(BaseTextClassification, Dataset):
    """ This represents a dataset that is of the form

    line1###label1

    line2###label2

    line3###label3
    .
    .
    .
    """

    def __init__(
        self, filename: str, tokenizers: Dict[str, BaseTokenizer] = WordTokenizer()
    ):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.lines, self.labels = self.get_lines_labels()

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

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value


class TextClassificationDatasetManager(DatasetsManager, ClassNursery):
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
        """

        Parameters
        ----------
        train_filename: str
            The path wehere the train file is stored
        dev_filename: str
            The path where the dev file is stored
        test_filename: str
            The path where the test file is stored
        tokenizers: Dict[str, BaseTokenizer]
            A mapping from namespace to the tokenizer
        namespace_vocab_options: Dict[str, Dict[str, Any]]
            A mapping from the name to options
        namespace_numericalizer_map: Dict[str, BaseNumericalizer]
            Every namespace can have a different numericalizer specified
        batch_size: int
            The batch size of the data returned
        """
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.tokenizers = tokenizers or {
            "tokens": WordTokenizer(),
            "char_tokens": CharacterTokenizer(),
        }
        self.namespace_vocab_options = namespace_vocab_options or {
            "char_tokens": {
                "start_token": " ",
                "end_token": " ",
                "pad_token": " ",
                "unk_token": " ",
            },
            "label": {"include_special_vocab": False},
        }
        self.namespace_numericalizer_map = namespace_numericalizer_map or {
            "tokens": Numericalizer(),
            "char_tokens": Numericalizer(),
        }
        self.namespace_numericalizer_map["label"] = Numericalizer()
        self.batch_size = batch_size

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
