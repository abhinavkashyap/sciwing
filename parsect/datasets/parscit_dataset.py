from typing import List, Dict, Union
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.tokenizers.character_tokenizer import CharacterTokenizer
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
        max_word_length: int,
        max_char_length: int,
        word_vocab_store_location: str,
        char_vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[int, None] = None,
        character_embedding_dimension: Union[int, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenizer=WordTokenizer(),
        word_tokenization_type="vanilla",
        word_add_start_end_token: bool = True,
        character_tokenizer=CharacterTokenizer(),
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
        :param max_word_length: type: int
        The maximum length after numericalization
        :param word_vocab_store_location: type: str
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
        :param word_embedding_type: type: str
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
        :param word_tokenizer
        The tokenizer that will be used to word_tokenize text
        :param word_tokenization_type: str
        Allowed type (vanilla, bert)
        Two types of tokenization are allowed. Either vanilla tokenization that is based on spacy.
        The default is WordTokenizer()
        If bert, then bert tokenization is performed and additional fields will be included in the output
        """

        super(ParscitDataset, self).__init__(
            filename=parscit_conll_file,
            dataset_type=dataset_type,
            max_num_words=max_num_words,
            max_length=max_word_length,
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
            character_tokenizer=character_tokenizer,
        )
        self.char_vocab_store_location = char_vocab_store_location
        self.character_embedding_dimension = character_embedding_dimension
        self.max_char_length = max_char_length
        self.word_add_start_end_token = word_add_start_end_token
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
        self.word_instances = self.word_tokenize(self.lines)
        self.word_vocab = Vocab(
            instances=self.word_instances,
            max_num_tokens=self.max_num_words,
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
        self.word_numericalizer = Numericalizer(vocabulary=self.word_vocab)

        # get character instances
        self.character_instances = self.character_tokenize(self.lines)
        self.char_vocab = Vocab(
            instances=self.character_instances,
            max_num_tokens=1e6,
            min_count=1,
            store_location=self.char_vocab_store_location,
            embedding_type="random",
            embedding_dimension=self.character_embedding_dimension,
            start_token=" ",
            end_token=" ",
            unk_token=" ",
            pad_token=" ",
        )
        self.char_vocab.build_vocab()

        # adding these to help conversion to characters later
        self.char_vocab.add_tokens(
            list(self.start_token)
            + list(self.end_token)
            + list(self.unk_token)
            + list(self.pad_token)
        )
        self.char_vocab.print_stats()
        self.char_numericalizer = Numericalizer(vocabulary=self.char_vocab)

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
        num_instances = len(self.word_instances)
        len_instances = [len(instance) for instance in self.word_instances]
        max_len_instance = max(len_instances)
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
        self.msg_printer.divider(f"Label Stats for Parscit {self.dataset_type} dataset")
        print(formatted)

        # print some other stats
        num_instances = len(self)
        other_stats_header = ["", "Value"]
        rows = [
            ("Num Instances", num_instances),
            ("Longest Instance Length", max_len_instance),
        ]

        other_stats_table = wasabi.table(
            data=rows, header=other_stats_header, divider=True
        )
        self.msg_printer.divider(f"Other stats for Parscit {self.dataset_type} dataset")
        print(other_stats_table)

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

    def get_preloaded_word_embedding(self):
        return self.word_vocab.load_embedding()

    def get_preloaded_char_embedding(self):
        return self.char_vocab.load_embedding(embedding_for="character")

    def __len__(self):
        return len(self.word_instances)

    def __getitem__(self, idx):
        word_instance = self.word_instances[idx]
        labels_string = self.labels[idx]
        labels_string = labels_string.split()
        len_instance = len(word_instance)

        # if instances are padded, then labels also have to be padded
        padded_word_instance = pack_to_length(
            tokenized_text=word_instance,
            max_length=self.max_length,
            pad_token=self.word_vocab.pad_token,
            add_start_end_token=self.word_add_start_end_token,
            start_token=self.word_vocab.start_token,
            end_token=self.word_vocab.end_token,
        )
        padded_labels = pack_to_length(
            tokenized_text=labels_string,
            max_length=self.max_length,
            pad_token="padding",
            add_start_end_token=self.word_add_start_end_token,
            start_token="starting",
            end_token="ending",
        )
        assert len(padded_word_instance) == len(padded_labels)

        tokens = self.word_numericalizer.numericalize_instance(padded_word_instance)
        padded_labels = [
            self.classnames2idx[self.normalized_classnames[label]]
            for label in padded_labels
        ]

        character_tokens = []
        # 1. For every word we get characters in the word
        # 2. Pad the characters to max_char_length
        # 3. Convert them into numbers
        # 4. Add them to character_tokens
        for word in padded_word_instance:
            character_instance = self.character_tokenizer.tokenize(word)
            padded_character_instance = pack_to_length(
                tokenized_text=character_instance,
                max_length=self.max_char_length,
                pad_token=" ",
                add_start_end_token=False,
            )
            padded_character_tokens = self.char_numericalizer.numericalize_instance(
                padded_character_instance
            )
            character_tokens.append(padded_character_tokens)

        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])
        label = torch.LongTensor(padded_labels)
        character_tokens = torch.LongTensor(
            character_tokens
        )  # max_word_len * max_char_len matrix

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "label": label,
            "instance": " ".join(padded_word_instance),
            "raw_instance": " ".join(word_instance),
            "char_tokens": character_tokens,
        }
        return instance_dict
