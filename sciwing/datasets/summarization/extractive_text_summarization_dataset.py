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


class ExtractiveSummarizationDataset(BaseAbstractiveTextSummarization, Dataset):
    """ This represents a dataset that is of the form
        sentence1\tsentence2\tsentence3###label1,label2label3###reference
        sentence1\tsentence2\tsentence3###label1,label2label3###reference
        sentence1\tsentence2\tsentence3###label1,label2label3###reference
        .
        .
        .
    """

    def __init__(self, filename: str, tokenizers: Dict[str, BaseTokenizer]):
        super().__init__(filename, tokenizers)
        self.filename = filename
        self.tokenizers = tokenizers
        self.docs, self.labels, self.refs = self.get_docs_labels_refs()

    def get_docs_labels_refs(self) -> (List[List[Line]], List[SeqLabel], List[Line]):
        docs: List[List[Line]] = []
        labels: List[SeqLabel] = []
        refs: List[Line] = []

        with open(self.filename, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not bool(line):
                    continue
                line_sents, line_labels, line_ref = line.strip().split("###")
                sents: List[str] = [sent.strip() for sent in line_sents.split("\t")]
                sents_labels: List[str] = [
                    sent_label.strip() for sent_label in line_labels.split(",")
                ]
                sents_refs: str = line_ref

                doc = [Line(text=sent, tokenizers=self.tokenizers) for sent in sents]
                label = SeqLabel(labels={"seq_label": sents_labels})
                ref = Line(text=sents_refs, tokenizers=self.tokenizers)
                docs.append(doc)
                labels.append(label)
                refs.append(ref)

        return docs, labels, refs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx) -> (List[Line], SeqLabel, Line):
        doc, label, ref = self.docs[idx], self.labels[idx], self.refs[idx]
        return doc, label, ref


class ExtractiveSummarizationDatasetManager(DatasetsManager):
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

        self.train_dataset = ExtractiveSummarizationDataset(
            filename=self.train_filename, tokenizers=self.tokenizers
        )

        self.dev_dataset = ExtractiveSummarizationDataset(
            filename=self.dev_filename, tokenizers=self.tokenizers
        )

        self.test_dataset = ExtractiveSummarizationDataset(
            filename=self.test_filename, tokenizers=self.tokenizers
        )

        super(ExtractiveSummarizationDatasetManager, self).__init__(
            train_dataset=self.train_dataset,
            dev_dataset=self.dev_dataset,
            test_dataset=self.test_dataset,
            namespace_vocab_options=self.namespace_vocab_options,
            namespace_numericalizer_map=self.namespace_numericalizer_map,
            batch_size=batch_size,
        )
