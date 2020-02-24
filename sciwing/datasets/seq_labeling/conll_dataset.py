from typing import List, Dict, Any
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.numericalizers.numericalizer import Numericalizer
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
from torch.utils.data import Dataset


class CoNLLDataset(BaseSeqLabelingDataset, Dataset):
    def __init__(
        self,
        filename: str,
        tokenizers: Dict[str, BaseTokenizer],
        column_names: List[str] = None,
    ):
        """ Dataset in CoNLL format

        Parameters
        ----------
        filename : str
            The CONLL filename
        tokenizers : Dict[str, BaseTokenizer]
        column_names: List[str]
            A list of column names for the three labels in the CoNLL format
            If this is not provided then we will use ["label_1", "label_2", "label_3"]
        """
        super().__init__(filename, tokenizers)
        if column_names is None:
            column_names = ["label_1", "label_2", "label_3"]
        assert len(column_names) == 3

        self.filename = filename
        self.tokenizers = tokenizers
        self.column_names = column_names
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[Line], List[SeqLabel]):
        lines: List[Line] = []
        labels: List[SeqLabel] = []
        with open(self.filename) as fp:
            lines_: List[str] = []
            labels_: List[List[str]] = []  # every list is a label for one namespace
            for text in fp:
                if bool(text.strip()):
                    text = text.strip()
                    line_labels = text.split()
                    line_ = line_labels[0]
                    label_ = line_labels[1:]
                    lines_.append(line_)
                    labels_.append(label_)
                else:
                    text_ = " ".join(lines_)
                    line, label = self._form_line_label(text=text_, labels=labels_)
                    lines.append(line)
                    labels.append(label)
                    lines_ = []
                    labels_ = []
            # handle the case when there is only one file without any new line
            else:
                if len(lines_) > 0 and len(lines) == 0:
                    text_ = " ".join(lines_)
                    line, label = self._form_line_label(text=text_, labels=labels_)
                    lines.append(line)
                    labels.append(label)

        return lines, labels

    def _form_line_label(self, text: str, labels: List[str]):
        line = Line(text=text, tokenizers=self.tokenizers)
        labels_ = zip(*labels)
        labels_ = zip(self.column_names, labels_)
        labels_ = dict(labels_)
        label = SeqLabel(labels=labels_)
        return line, label

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label = self.labels[idx]
        return line, label


class CoNLLDatasetManager(DatasetsManager, ClassNursery):
    def __init__(
        self,
        train_filename: str,
        dev_filename: str,
        test_filename: str,
        tokenizers: Dict[str, BaseTokenizer] = None,
        namespace_vocab_options: Dict[str, Dict[str, Any]] = None,
        namespace_numericalizer_map: Dict[str, BaseNumericalizer] = None,
        batch_size=10,
        column_names: List[str] = None,
    ):

        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.tokenizers = tokenizers or {
            "tokens": WordTokenizer(tokenizer="vanilla"),
            "char_tokens": CharacterTokenizer(),
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
            "char_tokens": Numericalizer(),
        }

        self.batch_size = batch_size

        if column_names is None:
            column_names = ["label_1", "label_2", "label_3"]

        for column_name in column_names:
            self.namespace_numericalizer_map[column_name] = Numericalizer()

        self.train_dataset = CoNLLDataset(
            filename=self.train_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        self.dev_dataset = CoNLLDataset(
            filename=self.dev_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        self.test_dataset = CoNLLDataset(
            filename=self.test_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        super(CoNLLDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )
