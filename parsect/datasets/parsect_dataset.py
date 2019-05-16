import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
import parsect.constants as constants
from parsect.utils.common import convert_secthead_to_json
from parsect.vocab.vocab import Vocab
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.numericalizer.numericalizer import Numericalizer

from wasabi import Printer

FILES = constants.FILES
SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']


class ParsectDataset(Dataset):
    def __init__(self,
                 secthead_label_file: str,
                 dataset_type: str,
                 max_num_words: int,
                 max_length: int,
                 vocab_store_location: str,
                 debug: bool = False,
                 debug_dataset_proportion=0.1):
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

        """
        self.dataset_type = dataset_type
        self.secthead_label_file = secthead_label_file
        self.max_num_words = max_num_words
        self.max_length = max_length
        self.store_location = vocab_store_location
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion

        self.word_tokenizer = WordTokenizer()
        self.label_mapping = self.get_label_mapping()
        self.allowable_dataset_types = ['train', 'valid', 'test']
        self.msg_printer = Printer()

        self.msg_printer.divider("{0} DATASET".format(self.dataset_type.upper()))

        assert self.dataset_type in self.allowable_dataset_types, "You can Pass one of these " \
                                                                  "for dataset types: {0}" \
            .format(self.allowable_dataset_types)

        self.parsect_json = convert_secthead_to_json(self.secthead_label_file)
        self.lines, self.labels = self.get_lines_labels()
        self.instances = self.tokenize(self.lines)

        self.vocab = Vocab(self.instances,
                           max_num_words=self.max_num_words,
                           store_location=self.store_location)
        self.vocab.build_vocab()

        self.numericalizer = Numericalizer(max_length=self.max_length,
                                           vocabulary=self.vocab)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx) -> (torch.LongTensor, torch.LongTensor):
        instance = self.instances[idx]

        # real length of tokens, numericalized tokens
        len_tokens, tokens = self.numericalizer.numericalize_instance(instance)
        label = self.labels[idx]
        label_idx = self.label_mapping[label]

        tokens = torch.LongTensor(tokens)
        label = torch.LongTensor([label_idx])
        len_tokens = torch.LongTensor([len_tokens])

        return tokens, label, len_tokens

    def get_lines_labels(self) -> (List[str], List[str]):
        """
        Returns the appropriate lines depending on the type of dataset
        :return:
        """
        texts = []
        labels = []
        parsect_json = self.parsect_json["parse_sect"]
        if self.dataset_type == 'train':
            parsect_json = filter(lambda json_line: json_line['file_no'] in [1, 20], parsect_json)

        elif self.dataset_type == 'valid':
            parsect_json = filter(lambda json_line: json_line['file_no'] in [21, 30], parsect_json)

        elif self.dataset_type == 'test':
            parsect_json = filter(lambda json_line: json_line['file_no'] in [30, 40], parsect_json)

        with self.msg_printer.loading("Loading"):
            for line_json in parsect_json:
                text = line_json['text']
                label = line_json['label']

                texts.append(text)
                labels.append(label)

        if self.debug:
            # randomly sample 10% samples and return
            num_text = len(texts)
            random_ints = np.random.randint(0, num_text - 1,
                                            size=int(self.debug_dataset_proportion * num_text))
            random_ints = list(random_ints)
            sample_texts = []
            sample_labels = []
            for random_int in random_ints:
                sample_texts.append(texts[random_int])
                sample_labels.append(texts[random_int])

        self.msg_printer.good('Finished Reading JSON lines from the data file')

        return texts, labels

    def tokenize(self,
                 lines: List[str]) -> List[List[str]]:
        """
        :param lines: type: List[str]
        These are text spans that will be tokenized
        :return: instances type: List[List[str]]
        """
        instances = self.word_tokenizer.tokenize_batch(lines)
        return instances

    @staticmethod
    def get_label_mapping() -> Dict[str, int]:
        categories = ['address', 'affiliation', 'author', 'bodyText',
                      'category', 'construct', 'copyright', 'email', 'equation',
                      'figure', 'figureCaption', 'footnote', 'keyword', 'listItem',
                      'note', 'page', 'reference', 'sectionHeader', 'subsectionHeader',
                      'subsubsectionHeader', 'tableCaption', 'table', 'title'
                      ]
        categories = [(word, idx) for idx, word in enumerate(categories)]
        categories = dict(categories)
        return categories

    def get_num_classes(self) -> int:
        return len(self.label_mapping.keys())


if __name__ == '__main__':
    import os

    vocab_store_location = os.path.join('.', 'vocab.json')
    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='train',
        max_num_words=1000,
        max_length=15,
        vocab_store_location=vocab_store_location
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='valid',
        max_num_words=1000,
        max_length=15,
        vocab_store_location=vocab_store_location
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='test',
        max_num_words=1000,
        max_length=15,
        vocab_store_location=vocab_store_location
    )
