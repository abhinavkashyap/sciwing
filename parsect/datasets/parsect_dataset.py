import torch
import wasabi
import numpy as np
import collections
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any
import parsect.constants as constants
from parsect.utils.common import convert_sectlabel_to_json
from parsect.utils.common import pack_to_length
from parsect.vocab.vocab import Vocab
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.numericalizer.numericalizer import Numericalizer
from sklearn.model_selection import StratifiedShuffleSplit
from wasabi import Printer
import numpy as np
from deprecated import deprecated

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


class ParsectDataset(Dataset):
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
        :param return_instances: type: bool
        If this is set, instead of numericalizing the instances,
        the instances themselves will be returned from __get_item__
        This is helpful in some cases like Elmo encoder that expect a list of sentences
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
        self.dataset_type = dataset_type
        self.secthead_label_file = secthead_label_file
        self.max_num_words = max_num_words
        self.max_length = max_length
        self.store_location = vocab_store_location
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion
        self.embedding_type = embedding_type
        self.embedding_dimension = embedding_dimension
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.tokenizer = tokenizer
        self.tokenization_type = tokenization_type
        self.classname2idx = self.get_label_mapping()
        self.idx2classname = {
            idx: classname for classname, idx in self.classname2idx.items()
        }
        self.allowable_dataset_types = ["train", "valid", "test"]
        self.msg_printer = Printer()

        self.msg_printer.divider("{0} DATASET".format(self.dataset_type.upper()))

        assert self.dataset_type in self.allowable_dataset_types, (
            "You can Pass one of these "
            "for dataset types: {0}".format(self.allowable_dataset_types)
        )

        self.parsect_json = convert_sectlabel_to_json(self.secthead_label_file)
        self.lines, self.labels = self.get_lines_labels_stratified()
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
    def get_lines_labels(self) -> (List[str], List[str]):
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

    def get_lines_labels_stratified(self) -> (List[str], List[str]):
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
        ) = self.get_train_valid_test_split(texts, labels)

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

    def tokenize(self, lines: List[str]) -> List[List[str]]:
        """
        :param lines: type: List[str]
        These are text spans that will be tokenized
        :return: instances type: List[List[str]]
        """
        instances = list(map(lambda line: self.tokenizer.tokenize(line), lines))
        return instances

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

    def get_train_valid_test_split(
        self, lines: List[str], labels: List[str]
    ) -> ((List[str], List[str]), (List[str], List[str]), (List[str], List[str])):
        len_lines = len(lines)
        len_labels = len(labels)

        assert len_lines == len_labels

        train_test_spliiter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=1729,
        )

        features = np.random.rand(len_lines)
        labels_idx_array = np.array([self.classname2idx[label] for label in labels])

        splits = list(train_test_spliiter.split(features, labels_idx_array))
        train_indices, test_valid_indices = splits[0]

        train_lines = [lines[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]

        test_valid_lines = [lines[idx] for idx in test_valid_indices]
        test_valid_labels = [labels[idx] for idx in test_valid_indices]

        validation_test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            train_size=1 - self.validation_size,
            random_state=1729,
        )

        len_test_valid_lines = len(test_valid_lines)
        len_test_valid_labels = len(test_valid_labels)

        assert len_test_valid_labels == len_test_valid_lines

        test_valid_features = np.random.rand(len_test_valid_lines)
        test_valid_labels_idx_array = np.array(
            [self.classname2idx[label] for label in test_valid_labels]
        )

        test_valid_splits = list(
            validation_test_splitter.split(
                test_valid_features, test_valid_labels_idx_array
            )
        )
        test_indices, validation_indices = test_valid_splits[0]

        test_lines = [test_valid_lines[idx] for idx in test_indices]
        test_labels = [test_valid_labels[idx] for idx in test_indices]

        validation_lines = [test_valid_lines[idx] for idx in validation_indices]
        validation_labels = [test_valid_labels[idx] for idx in validation_indices]

        return (
            (train_lines, train_labels),
            (validation_lines, validation_labels),
            (test_lines, test_labels),
        )


if __name__ == "__main__":
    import os
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    vocab_store_location = os.path.join(".", "vocab.json")
    DEBUG = True
    MAX_NUM_WORDS = 500
    MAX_LENGTH = 10
    DEBUG_DATASET_PROPORTION = 0.1

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        train_size=0.8,
        test_size=0.2,
        validation_size=0.5,
        tokenization_type="bert",
        tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
        start_token="[CLS]",
        end_token="[SEP]",
        pad_token="[PAD]",
    )

    train_dataset.get_stats()

    idx = 1000
    print(f"bert tokens: {train_dataset[idx]['bert_tokens']}")
    print(f"raw instance: {train_dataset[idx]['raw_instance']}")
    print(f"segment ids: {train_dataset[idx]['segment_ids']}")

    os.remove(vocab_store_location)
