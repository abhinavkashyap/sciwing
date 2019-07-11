from typing import List, Dict, Union
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.datasets.TextClassificationDataset import TextClassificationDataset
from torch.utils.data import Dataset
import wasabi
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.utils.common import pack_to_length
import numpy as np
import torch
import collections


class ParscitDataset(Dataset, TextClassificationDataset):
    def __init__(
        self,
        parscit_conll_file: str,
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
        add_start_end_token: bool = True,
    ):
        """

        :param parscit_conll_file: type: str
        The parscit file written in the conll format
        The conll format for parscit consists of a line
        word label label label
        We cosndier only the word and the last label
        Citation strings are separated by a new line
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

        super(ParscitDataset, self).__init__(
            filename=parscit_conll_file,
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
        self.add_start_end_token = add_start_end_token
        self.normalized_classnames = {
            "author": "author",
            "booktitle": "booktitle",
            "date": "date",
            "editor": "editor",
            "institution": "institution",
            "journal": "journal",
            "location": "location",
            "note": "note",
            "notes": "note",
            "pages": "pages",
            "publisher": "publisher",
            "tech": "tech",
            "title": "title",
            "volume": "volume",
            "padding": "padding",
            "starting": "starting",
            "ending": "ending",
        }
        self.classnames2idx = self.get_classname2idx()
        self.idx2classname = {
            idx: classname for classname, idx in self.classnames2idx.items()
        }
        self.msg_printer = wasabi.Printer()
        self.lines, self.labels = self.get_lines_labels()
        self.instances = self.tokenize(self.lines)
        self.word_vocab = Vocab(
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
        self.word_vocab.build_vocab()
        self.word_vocab.print_stats()
        self.numericalizer = Numericalizer(vocabulary=self.word_vocab)

    @staticmethod
    def get_classname2idx() -> Dict[str, int]:
        categories = [
            "author",
            "booktitle",
            "date",
            "editor",
            "institution",
            "journal",
            "location",
            "note",
            "pages",
            "publisher",
            "tech",
            "title",
            "volume",
            "padding",  # adding to give an extra class name to padded tokens
            "starting",
            "ending",
        ]

        categories = [(category, idx) for idx, category in enumerate(categories)]
        categories = dict(categories)
        return categories

    def get_num_classes(self) -> int:
        return len(self.classnames2idx.keys())

    def get_class_names_from_indices(self, indices: List[int]):
        return [self.idx2classname[idx] for idx in indices]

    def get_disp_sentence_from_indices(self, indices: List[int]) -> str:
        token = [
            self.word_vocab.get_token_from_idx(idx)
            for idx in indices
            if idx != self.word_vocab.special_vocab[self.word_vocab.pad_token][1]
        ]
        sentence = " ".join(token)
        return sentence

    def get_stats(self):
        num_instances = len(self.instances)
        all_labels = []
        for idx in range(num_instances):
            iter_dict = self[idx]
            labels = iter_dict["label"]
            all_labels.extend(labels.cpu().numpy().tolist())

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
            f"Number of instances in {self.dataset_type} dataset - {len(self)}"
        )

    def get_lines_labels(self) -> (List[str], List[str]):
        """
        Returns citation string, label
        citation string is a space separated string
        label is a space separated string of different labels
        :return:
        """
        lines = []
        labels = []
        with open(self.filename) as fp:
            words = []
            tags = []
            for line in fp:
                if bool(line.strip()):
                    citation_word_tag = line.split()
                    word, tag = citation_word_tag[0], citation_word_tag[-1]
                    words.append(word)
                    tags.append(tag)
                else:
                    line = " ".join(words)
                    label = " ".join(tags)

                    # make sure to have as many tags as words
                    assert len(words) == len(tags)
                    words = []
                    tags = []
                    lines.append(line)
                    labels.append(label)

        if self.debug:
            # randomly sample `self.debug_dataset_proportion`  samples and return
            num_lines = len(lines)
            np.random.seed(1729)  # so we can debug deterministically
            random_ints = np.random.randint(
                0, num_lines - 1, size=int(self.debug_dataset_proportion * num_lines)
            )
            random_ints = list(random_ints)
            sample_lines = []
            sample_labels = []
            for random_int in random_ints:
                sample_lines.append(lines[random_int])
                sample_labels.append(labels[random_int])
            lines = sample_lines
            labels = sample_labels

        return lines, labels

    def get_preloaded_embedding(self):
        return self.word_vocab.load_embedding()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        labels_string = self.labels[idx]
        labels_string = labels_string.split()
        len_instance = len(instance)

        # if instances are padded, then labels also have to be padded
        padded_instance = pack_to_length(
            tokenized_text=instance,
            max_length=self.max_length,
            pad_token=self.word_vocab.pad_token,
            add_start_end_token=self.add_start_end_token,
            start_token=self.word_vocab.start_token,
            end_token=self.word_vocab.end_token,
        )
        padded_labels = pack_to_length(
            tokenized_text=labels_string,
            max_length=self.max_length,
            pad_token="padding",
            add_start_end_token=self.add_start_end_token,
            start_token="starting",
            end_token="ending",
        )
        assert len(padded_instance) == len(padded_labels)

        tokens = self.numericalizer.numericalize_instance(padded_instance)
        padded_labels = [
            self.classnames2idx[self.normalized_classnames[label]]
            for label in padded_labels
        ]

        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])
        label = torch.LongTensor(padded_labels)

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "label": label,
            "instance": " ".join(padded_instance),
            "raw_instance": " ".join(instance),
        }
        return instance_dict
