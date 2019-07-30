import torch
import wasabi
import collections
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any, Optional
import parsect.constants as constants
from parsect.utils.common import convert_sectlabel_to_json
from parsect.utils.common import pack_to_length
from parsect.vocab.vocab import Vocab
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.numericalizer.numericalizer import Numericalizer
from wasabi import Printer
import numpy as np
from deprecated import deprecated
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ParsectDataset(Dataset, BaseTextClassification):
    """Parsect dataset consists of dataset for logical classification of scientific papers

    """

    def __init__(
        self,
        secthead_label_file: str,
        dataset_type: str,
        max_num_words: int,
        max_length: int,
        word_vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[int, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenizer=WordTokenizer(),
        word_tokenization_type="vanilla",
    ):
        super(ParsectDataset, self).__init__(
            filename=secthead_label_file,
            dataset_type=dataset_type,
            max_num_words=max_num_words,
            max_length=max_length,
            word_vocab_store_location=word_vocab_store_location,
            debug=debug,
            debug_dataset_proportion=debug_dataset_proportion,
            word_embedding_type=word_embedding_type,
            word_embedding_dimension=word_embedding_dimension,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
            train_size=train_size,
            test_size=test_size,
            validation_size=validation_size,
            word_tokenizer=word_tokenizer,
            word_tokenization_type=word_tokenization_type,
        )
        self.classname2idx = self.get_classname2idx()
        self.idx2classname = {
            idx: classname for classname, idx in self.classname2idx.items()
        }

        self.msg_printer = Printer()

        self.parsect_json = convert_sectlabel_to_json(self.filename)
        self.lines, self.labels = self.get_lines_labels()
        self.instances = self.word_tokenize(self.lines)

        self.vocab = Vocab(
            instances=self.instances,
            max_num_tokens=self.max_num_words,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            start_token=self.start_token,
            end_token=self.end_token,
            store_location=self.store_location,
            embedding_type=self.embedding_type,
            embedding_dimension=self.embedding_dimension,
        )
        self.vocab.build_vocab()
        self.vocab.print_stats()

        self.numericalizer = Numericalizer(vocabulary=self.vocab)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx) -> Dict[str, Any]:
        line = self.lines[idx]
        label = self.labels[idx]

        return self.get_iter_dict(
            line=line,
            word_vocab=self.vocab,
            word_tokenizer=self.word_tokenizer,
            max_word_length=self.max_length,
            word_add_start_end_token=True,
            labels=label,
        )

    @deprecated(reason="Deprecated because of bad train-valid-test split.")
    def get_lines_labels_deprecated(self) -> (List[str], List[str]):
        """
        Returns the appropriate lines depending on the type of dataset
        :return:
        """
        texts = []
        labels = []
        parsect_json = self.parsect_json["parse_sect"]
        if self.dataset_type == "train":
            parsect_json = filter(
                lambda json_line: json_line["file_no"] in list(range(1, 21)),
                parsect_json,
            )

        elif self.dataset_type == "valid":
            parsect_json = filter(
                lambda json_line: json_line["file_no"] in list(range(21, 31)),
                parsect_json,
            )

        elif self.dataset_type == "test":
            parsect_json = filter(
                lambda json_line: json_line["file_no"] in list(range(31, 41)),
                parsect_json,
            )

        with self.msg_printer.loading("Loading"):
            for line_json in parsect_json:
                text = line_json["text"]
                label = line_json["label"]

                texts.append(text)
                labels.append(label)

        if self.debug:
            # randomly sample 10% samples and return
            num_text = len(texts)
            np.random.seed(1729)  # so we can debug deterministically
            random_ints = np.random.randint(
                0, num_text - 1, size=int(self.debug_dataset_proportion * num_text)
            )
            random_ints = list(random_ints)
            sample_texts = []
            sample_labels = []
            for random_int in random_ints:
                sample_texts.append(texts[random_int])
                sample_labels.append(labels[random_int])
            texts = sample_texts
            labels = sample_labels

        self.msg_printer.good("Finished Reading JSON lines from the data file")

        return texts, labels

    def get_lines_labels(self) -> (List[str], List[str]):
        texts = []
        labels = []
        parsect_json = self.parsect_json["parse_sect"]

        for line_json in parsect_json:
            text = line_json["text"]
            label = line_json["label"]

            texts.append(text)
            labels.append(label)

        (train_lines, train_labels), (validation_lines, validation_labels), (
            test_lines,
            test_labels,
        ) = self.get_train_valid_test_stratified_split(
            texts, labels, self.classname2idx
        )

        if self.dataset_type == "train":
            texts = train_lines
            labels = train_labels
        elif self.dataset_type == "valid":
            texts = validation_lines
            labels = validation_labels
        elif self.dataset_type == "test":
            texts = test_lines
            labels = test_labels

        if self.debug:
            # randomly sample `self.debug_dataset_proportion`  samples and return
            num_text = len(texts)
            np.random.seed(1729)  # so we can debug deterministically
            random_ints = np.random.randint(
                0, num_text - 1, size=int(self.debug_dataset_proportion * num_text)
            )
            random_ints = list(random_ints)
            sample_texts = []
            sample_labels = []
            for random_int in random_ints:
                sample_texts.append(texts[random_int])
                sample_labels.append(labels[random_int])
            texts = sample_texts
            labels = sample_labels

        return texts, labels

    @classmethod
    def get_classname2idx(cls) -> Dict[str, int]:
        categories = [
            "address",
            "affiliation",
            "author",
            "bodyText",
            "category",
            "construct",
            "copyright",
            "email",
            "equation",
            "figure",
            "figureCaption",
            "footnote",
            "keyword",
            "listItem",
            "note",
            "page",
            "reference",
            "sectionHeader",
            "subsectionHeader",
            "subsubsectionHeader",
            "tableCaption",
            "table",
            "title",
        ]
        categories = [(word, idx) for idx, word in enumerate(categories)]
        categories = dict(categories)
        return categories

    def get_num_classes(self) -> int:
        return len(self.classname2idx.keys())

    def get_class_names_from_indices(self, indices: List) -> List[str]:
        return [self.idx2classname[idx] for idx in indices]

    def get_disp_sentence_from_indices(self, indices: List) -> str:
        pad_token_index = self.vocab.special_vocab[self.vocab.pad_token][1]
        start_token_index = self.vocab.special_vocab[self.vocab.start_token][1]
        end_token_index = self.vocab.special_vocab[self.vocab.end_token][1]
        special_indices = [pad_token_index, start_token_index, end_token_index]

        token = [
            self.vocab.get_token_from_idx(idx)
            for idx in indices
            if idx not in special_indices
        ]
        sentence = " ".join(token)
        return sentence

    def print_stats(self):
        num_instances = len(self.instances)
        all_labels = []
        for idx in range(num_instances):
            iter_dict = self[idx]
            labels = iter_dict["label"]
            all_labels.append(labels.item())

        labels_stats = dict(collections.Counter(all_labels))
        classes = list(set(labels_stats.keys()))
        classes = sorted(classes)
        header = ["label index", "label name", "count"]
        rows = [
            (class_, self.idx2classname[class_], labels_stats[class_])
            for class_ in classes
        ]
        formatted = wasabi.table(data=rows, header=header, divider=True)
        self.msg_printer.divider("Stats for {0} dataset".format(self.dataset_type))
        print(formatted)
        self.msg_printer.info(
            "Number of instances in {0} dataset - {1}".format(
                self.dataset_type, len(self)
            )
        )

    def get_preloaded_word_embedding(self) -> torch.FloatTensor:
        return self.vocab.load_embedding()

    def emits_keys(cls):
        return {
            "tokens": f"A torch.LongTensor of size `max_length`. "
            f"Example [0, 0, 1, 100] where every number represents an index in the vocab",
            "len_tokens": f"A torch.LongTensor. "
            f"Example [2] representing the number of tokens without padding",
            "label": f"A torch.LongTensor representing the label corresponding to the "
            f"instance. Example [2] representing class 2",
            "instance": f"A string that is padded to ``max_length``.",
            "raw_instance": f"A string that is not padded",
        }

    @classmethod
    def get_iter_dict(
        cls,
        line: str,
        word_vocab: Vocab,
        word_tokenizer: WordTokenizer,
        max_word_length: int,
        word_add_start_end_token: bool,
        labels: Optional[str] = None,
    ):
        word_instance = word_tokenizer.tokenize(line)
        len_instance = len(word_instance)
        word_numericalizer = Numericalizer(vocabulary=word_vocab)
        classnames2idx = ParsectDataset.get_classname2idx()

        if labels is not None:
            assert len_instance == len(word_instance)
            labels = classnames2idx[labels]
            label = torch.LongTensor([labels])

        padded_instance = pack_to_length(
            tokenized_text=word_instance,
            max_length=max_word_length,
            pad_token=word_vocab.pad_token,
            add_start_end_token=True,  # TODO: remove hard coded value here
            start_token=word_vocab.start_token,
            end_token=word_vocab.end_token,
        )

        tokens = word_numericalizer.numericalize_instance(padded_instance)

        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "instance": " ".join(padded_instance),
            "raw_instance": " ".join(word_instance),
        }

        if labels is not None:
            instance_dict["label"] = label

        return instance_dict
