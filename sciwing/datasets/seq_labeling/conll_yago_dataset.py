from sciwing.data.contextual_lines import LineWithContext
from sciwing.data.seq_label import SeqLabel
from sciwing.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
from sciwing.numericalizers.numericalizer import Numericalizer
from torch.utils.data import Dataset
from typing import Dict, List, Any
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.numericalizers.base_numericalizer import BaseNumericalizer
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager
import copy


class ConllYagoDataset(BaseSeqLabelingDataset, Dataset):
    def __init__(
        self,
        filename: str,
        tokenizers: Dict[str, BaseTokenizer],
        column_names: List[str] = None,
    ):
        """

        Parameters
        ----------
        filename
        tokenizers : Dict[str, BaseTokenizer]
            A mapping between
        column_names : List[str]
        Maximum one column for NER.
        """
        super().__init__(filename, tokenizers)
        if column_names is None:
            column_names = ["NER"]
        assert len(column_names) == 1
        self.filename = filename
        self.tokenizers = tokenizers
        self.column_names = column_names
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[LineWithContext], List[SeqLabel]):
        lines: List[LineWithContext] = []
        labels: List[SeqLabel] = []

        with open(self.filename) as fp:
            words_: List[str] = []
            labels_: List[str] = []
            yago_entities: List[str] = []

            for line in fp:
                line_ = line.strip()
                if bool(line_):
                    line_labels = line_.split()
                    word = line_labels[0]
                    ner_label = line_labels[1]
                    yago_entity = line_labels[2]

                    if yago_entity != "None":
                        yago_entity = yago_entity.split("_")
                        yago_entity = " ".join(yago_entity)
                        yago_entities.append(yago_entity)

                    words_.append(word)
                    labels_.append(ner_label)

                elif "DOCSTART" in line_:
                    continue
                else:
                    if len(words_) > 0 and len(labels_) > 0:
                        text = " ".join(words_)

                        if len(yago_entities) == 0:
                            yago_entities = ["NULL"]

                        line, label = self._form_line_label(
                            line=text, label=labels_, yago_entities=yago_entities
                        )
                        words_: List[str] = []
                        labels_: List[str] = []
                        yago_entities: List[str] = []
                        lines.append(line)
                        labels.append(label)

        return lines, labels

    def _form_line_label(self, line: str, label: List[str], yago_entities: List[str]):
        line = LineWithContext(
            text=line, context=yago_entities, tokenizers=self.tokenizers
        )
        label = SeqLabel({self.column_names[0]: label})
        return line, label

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label = self.labels[idx]
        return line, label


class ConllYagoDatasetsManager(DatasetsManager, ClassNursery):
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

        namespace_vocab_options_defaults = {
            "char_tokens": {
                "start_token": " ",
                "end_token": " ",
                "pad_token": " ",
                "unk_token": " ",
            }
        }

        if namespace_vocab_options is None:
            namespace_vocab_options = {}

        self.namespace_vocab_options = copy.deepcopy(namespace_vocab_options_defaults)

        for namespace, options in self.namespace_vocab_options.items():
            user_passed = namespace_vocab_options.get(namespace, {})
            self.namespace_vocab_options[namespace] = {**options, **user_passed}

        self.namespace_numericalizer_map = namespace_numericalizer_map or {
            "tokens": Numericalizer(),
            "char_tokens": Numericalizer(),
        }

        self.batch_size = batch_size

        if column_names is None:
            column_names = ["NER"]

        for column_name in column_names:
            self.namespace_numericalizer_map[column_name] = Numericalizer()

        self.train_dataset = ConllYagoDataset(
            filename=self.train_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        self.dev_dataset = ConllYagoDataset(
            filename=self.dev_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        self.test_dataset = ConllYagoDataset(
            filename=self.test_filename,
            tokenizers=self.tokenizers,
            column_names=column_names,
        )

        super(ConllYagoDatasetsManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )
