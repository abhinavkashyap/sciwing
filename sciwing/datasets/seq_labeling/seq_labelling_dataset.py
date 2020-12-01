from sciwing.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
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


class SeqLabellingDataset(BaseSeqLabelingDataset, Dataset):
    """ This represents a dataset that is of the form

        word1###label1 word2###label2 word3###label3

        word1###label1 word2###label2 word3###label3

        word1###label1 word2###label2 word3###label3

        .

        .

        .
    """

    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[Line], List[SeqLabel]):
        lines: List[Line] = []
        labels: List[SeqLabel] = []

        with open(self.filename, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not bool(line):
                    continue
                lines_and_labels = line.strip().split(" ")
                words: List[str] = []
                word_labels: List[str] = []
                for word_line_labels in lines_and_labels:
                    word, word_label = word_line_labels.split("###")
                    word = word.strip()
                    word_label = word_label.strip()
                    words.append(word)
                    word_labels.append(word_label)

                line = Line(text=" ".join(words), tokenizers=self.tokenizers)
                label = SeqLabel(labels={"seq_label": word_labels})
                lines.append(line)
                labels.append(label)

        return lines, labels

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (Line, SeqLabel):
        line, label = self.lines[idx], self.labels[idx]
        return line, label


class SeqLabellingDatasetManager(DatasetsManager):
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
        self.namespace_numericalizer_map["seq_label"] = Numericalizer()

        self.batch_size = batch_size

        self.train_dataset = SeqLabellingDataset(
            filename=self.train_filename, tokenizers=self.tokenizers
        )

        self.dev_dataset = SeqLabellingDataset(
            filename=self.dev_filename, tokenizers=self.tokenizers
        )

        self.test_dataset = SeqLabellingDataset(
            filename=self.test_filename, tokenizers=self.tokenizers
        )

        super(SeqLabellingDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )
