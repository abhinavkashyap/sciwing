import torch
import wasabi
import collections
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any
import parsect.constants as constants
from parsect.utils.common import convert_sectlabel_to_json
from parsect.utils.common import pack_to_length
from parsect.vocab.vocab import Vocab
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.numericalizer.numericalizer import Numericalizer
from wasabi import Printer
import numpy as np
from deprecated import deprecated
from parsect.datasets.TextClassificationDataset import TextClassificationDataset

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ParsectDataset(Dataset, TextClassificationDataset):
    def __init__(
        self,
        secthead_label_file: str,
        dataset_type: str,
        max_num_words: int,
        max_length: int,
        vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        embedding_type: Union[str, None] = None,
        embedding_dimension: Union[int, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        tokenizer=WordTokenizer(),
        tokenization_type="vanilla",
    ):
        """
        :param dataset_type: type: str
        One of ['train', 'valid', 'test']
        :param max_num_words: type: int
        The top frequent `max_num_words` to consider
        :param max_length: type: int
        The maximum length after numericalization
        :param vocab_store_location: type: str
        The vocab store location to store vocabulary
        This should be a json filename
        :param debug: type: bool
        If debug is true, then we randomly sample
        10% of the dataset and work with it. This is useful
        for faster automated tests and looking at random
        examples
        :param debug_dataset_proportion: type: float
        Send a number (0.0, 1.0) and a random proportion of the dataset
        will be used for debug purposes
        :param embedding_type: type: str
        Pre-loaded embedding type to load.
        :param start_token: type: str
        The start token is the token appended to the beginning of the list of tokens
        :param end_token: type: str
        The end token is the token appended to the end of the list of tokens
        :param pad_token: type: str
        The pad token is used when the length of the input is less than maximum length
        :param unk_token: type: str
        unk is the token that is used when the word is OOV
        :param train_size: float
        The proportion of the dataset that is used for training
        :param test_size: float
        The proportion of the dataset that is used for testing
        :param validation_size: float
        The proportion of the test dataset that is used for validation
        :param tokenizer
        The tokenizer that will be used to tokenize text
        :param tokenization_type: str
        Allowed type (vanilla, bert)
        Two types of tokenization are allowed. Either vanilla tokenization that is based on spacy.
        The default is WordTokenizer()
        If bert, then bert tokenization is performed and additional fields will be included in the output
        """
        super(ParsectDataset, self).__init__(
            filename=secthead_label_file,
            dataset_type=dataset_type,
            max_num_words=max_num_words,
            max_length=max_length,
            vocab_store_location=vocab_store_location,
            debug=debug,
            debug_dataset_proportion=debug_dataset_proportion,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
            train_size=train_size,
            test_size=test_size,
            validation_size=validation_size,
            tokenizer=tokenizer,
            tokenization_type=tokenization_type,
        )
        self.classname2idx = self.get_label_mapping()
        self.idx2classname = {
            idx: classname for classname, idx in self.classname2idx.items()
        }

        self.msg_printer = Printer()

        self.parsect_json = convert_sectlabel_to_json(self.filename)
        self.lines, self.labels = self.get_lines_labels()
        self.instances = self.tokenize(self.lines)

        self.vocab = Vocab(
            instances=self.instances,
            max_num_words=self.max_num_words,
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
        instance = self.instances[idx]
        label = self.labels[idx]
        label_idx = self.classname2idx[label]
        len_instance = len(instance)

        padded_instance = pack_to_length(
            tokenized_text=instance,
            max_length=self.max_length,
            pad_token=self.vocab.pad_token,
            add_start_end_token=True,  # TODO: remove hard coded value here
            start_token=self.vocab.start_token,
            end_token=self.vocab.end_token,
        )

        tokens = self.numericalizer.numericalize_instance(padded_instance)

        bert_tokens = -1  # -1 indicates no bert tokens
        segment_ids = -1  # -1 indicates no bert tokens

        if self.tokenization_type == "bert":
            bert_tokens = self.tokenizer.convert_tokens_to_ids(padded_instance)
            segment_ids = [0] * len(padded_instance)
            bert_tokens = torch.LongTensor(bert_tokens)
            segment_ids = torch.LongTensor(segment_ids)

        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])
        label = torch.LongTensor([label_idx])

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "label": label,
            "instance": " ".join(padded_instance),
            "raw_instance": " ".join(instance),
            "bert_tokens": bert_tokens,
            "segment_ids": segment_ids,
        }

        return instance_dict

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

    @staticmethod
    def get_label_mapping() -> Dict[str, int]:
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

    def get_class_names_from_indices(self, indices: List):
        return [self.idx2classname[idx] for idx in indices]

    def get_disp_sentence_from_indices(self, indices: List) -> str:

        token = [
            self.vocab.get_token_from_idx(idx)
            for idx in indices
            if idx != self.vocab.special_vocab[self.vocab.pad_token][1]
        ]
        sentence = " ".join(token)
        return sentence

    def get_stats(self):
        """
        Return some stats about the dataset
        """
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

    def get_preloaded_embedding(self) -> torch.FloatTensor:
        return self.vocab.load_embedding()
