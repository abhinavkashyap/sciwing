from torch.utils.data import Dataset
from typing import Union, Dict, List, Any
from parsect.utils.common import convert_generic_sect_to_json
import wasabi
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.utils.common import pack_to_length
import torch
import collections


class GenericSectDataset(Dataset):
    def __init__(
        self,
        generic_sect_filename: str,
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
        add_start_end_token: bool = True,
    ):
        super(GenericSectDataset, self).__init__()
        self.generic_sect_filename = generic_sect_filename
        self.dataset_type = dataset_type
        self.max_num_words = max_num_words
        self.max_length = max_length
        self.vocab_store_location = vocab_store_location
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
        self.add_start_end_token = add_start_end_token
        self.msg_printer = wasabi.Printer()

        self.word_tokenizer = WordTokenizer()

        self.allowable_dataset_types = ["train", "valid", "test"]
        assert (
            self.dataset_type in self.allowable_dataset_types
        ), f"You can pass one of these for dataset type {self.allowable_dataset_types}"

        self.label2idx = self.get_label_mapping()
        self.idx2label = {idx: class_name for class_name, idx in self.label2idx.items()}

        self.generic_sect_json = convert_generic_sect_to_json(
            self.generic_sect_filename
        )
        self.headers, self.labels = self.get_header_labels()
        self.instances = self.tokenize(self.headers)

        self.vocab = Vocab(
            instances=self.instances,
            max_num_words=self.max_num_words,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            start_token=self.start_token,
            end_token=self.end_token,
            store_location=self.vocab_store_location,
            embedding_type=self.embedding_type,
            embedding_dimension=self.embedding_dimension,
        )
        self.vocab.build_vocab()
        self.vocab.print_stats()

        self.numericalizer = Numericalizer(self.vocab)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        instance = self.instances[idx]
        label = self.labels[idx]
        label_idx = self.label2idx[label]
        len_instance = len(instance)

        padded_instance = pack_to_length(
            tokenized_text=instance,
            max_length=self.max_length,
            pad_token=self.pad_token,
            add_start_end_token=self.add_start_end_token,
            start_token=self.start_token,
            end_token=self.end_token,
        )

        tokens = self.numericalizer.numericalize_instance(padded_instance)
        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])
        label = torch.LongTensor([label_idx])

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "label": label,
            "instance": " ".join(padded_instance),
            "raw_instance": " ".join(instance),
        }

        return instance_dict

    def get_header_labels(self) -> (List[str], List[str]):
        headers = []
        labels = []

        generic_sect_json = self.generic_sect_json["generic_sect"]
        for line in generic_sect_json:
            label = line["label"]
            header = line["header"]
            label = label.strip()
            header = header.strip()
            headers.append(header)
            labels.append(label)

        (train_headers, train_labels), (valid_headers, valid_labels), (
            test_headers,
            test_labels,
        ) = self.get_train_valid_test_split(headers, labels)

        if self.dataset_type == "train":
            return train_headers, train_labels
        elif self.dataset_type == "valid":
            return valid_headers, valid_labels
        elif self.dataset_type == "test":
            return test_headers, test_labels

    def get_preloaded_embedding(self) -> torch.FloatTensor:
        return self.vocab.load_embedding()

    @staticmethod
    def get_label_mapping() -> Dict[str, int]:
        categories = [
            "abstract",
            "categories-and-subject-descriptors",
            "general-terms",
            "introduction",
            "background",
            "related-works",
            "method",
            "evaluation",
            "discussions",
            "conclusions",
            "acknowledgments",
            "references",
        ]

        categories = [(category, idx) for idx, category in enumerate(categories)]
        categories = dict(categories)
        return categories

    def get_train_valid_test_split(
        self, headers: List[str], labels: List[str]
    ) -> ((List[str], List[str]), (List[str], List[str]), (List[str], List[str])):

        len_headers = len(headers)
        len_labels = len(labels)

        assert len_headers == len_labels

        train_test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=1729,
        )
        features = np.random.rand(len_headers)
        labels_idx_array = np.array([self.label2idx[label] for label in labels])

        splits = list(train_test_splitter.split(features, labels_idx_array))
        train_indices, test_valid_indices = splits[0]

        train_headers = [headers[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]

        test_valid_headers = [headers[idx] for idx in test_valid_indices]
        test_valid_labels = [labels[idx] for idx in test_valid_indices]

        # further split into validation set
        validation_test_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            train_size=1 - self.validation_size,
            random_state=1729,
        )
        len_test_valid_headers = len(test_valid_headers)
        len_test_valid_labels = len(test_valid_labels)
        assert len_test_valid_headers == len_test_valid_labels

        test_valid_features = np.random.rand(len_test_valid_labels)
        test_valid_labels_idx_array = np.array(
            [self.label2idx[label] for label in test_valid_labels]
        )

        test_valid_splits = list(
            validation_test_splitter.split(
                test_valid_features, test_valid_labels_idx_array
            )
        )
        test_indices, validation_indices = test_valid_splits[0]

        test_headers = [test_valid_headers[idx] for idx in test_indices]
        test_labels = [test_valid_labels[idx] for idx in test_indices]

        validation_headers = [test_valid_headers[idx] for idx in validation_indices]
        validation_labels = [test_valid_labels[idx] for idx in validation_indices]

        return (
            (train_headers, train_labels),
            (validation_headers, validation_labels),
            (test_headers, test_labels),
        )

    def tokenize(self, headers: List[str]) -> List[List[str]]:
        """

        :param headers: type: List[str]
        Header strings that will be tokenized
        :return: instances type: List[List[str]]
        """
        instances = self.word_tokenizer.tokenize_batch(headers)
        return instances

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
            (class_, self.idx2label[class_], labels_stats[class_]) for class_ in classes
        ]
        formatted = wasabi.table(data=rows, header=header, divider=True)
        self.msg_printer.divider("Stats for {0} dataset".format(self.dataset_type))
        print(formatted)
        self.msg_printer.info(
            "Number of instances in {0} dataset - {1}".format(
                self.dataset_type, len(self)
            )
        )

    def get_num_classes(self):
        return len(self.label2idx.keys())

    def get_class_names_from_indices(self, indices: List):
        return [self.idx2label[idx] for idx in indices]

    def get_disp_sentence_from_indices(self, indices: List) -> str:
        token = [
            self.vocab.get_token_from_idx(idx)
            for idx in indices
            if idx != self.vocab.special_vocab[self.vocab.pad_token]
        ]
        sentence = " ".join(token)
        return sentence
