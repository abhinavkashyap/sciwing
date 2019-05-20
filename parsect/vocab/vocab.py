from typing import List, Dict, Tuple, Any
from collections import Counter
from operator import itemgetter
import json
import os
from wasabi import Printer
import wasabi


class Vocab:
    def __init__(self,
                 instances: List[List[str]],
                 max_num_words: int,
                 min_count: int = 1,
                 unk_token: str = '<UNK>',
                 pad_token: str = '<PAD>',
                 start_token: str = '<SOS>',
                 end_token: str = '<EOS>',
                 special_token_freq: float = 1e10,
                 store_location: str = None
                 ):
        """

        :param instances: type: List[List[str]]
         Pass in the list of tokenized instances from which vocab is built
        :param max_num_words: type: int
        The top `max_num_words` frequent words will be considered for
        vocabulary and the rest of them will be mapped to `unk_token`
        :param min_count: type: int
        All words that do not have min count will be mapped to `unk_token`
        :param unk_token: str
        This token will be used for unknown words
        :param pad_token: type: str
        This token will be used for <PAD> words
        :param start_token: type: str
        This token will be used for start of sentence indicator
        :param end_token: type: str
        This token will be used for end of sentence indicator
        :param special_token_freq: type: float
        special tokens should have high frequency.
        The higher the frequency, the more common they are
        :param store_location: type: str
        The users can provide a store location optionally.
        The vocab will be stored in the location
        If the file exists then, the vocab will be restored from the file, rather than building it.
        """
        self.instances = instances
        self.max_num_words = max_num_words
        self.min_count = min_count
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.special_token_freq = special_token_freq
        self.vocab = None
        self.orig_vocab = None
        self.idx2token = None
        self.token2idx = None
        self.store_location = store_location
        self.msg_printer = Printer()

        # store the special tokens
        self.special_vocab = {
            self.unk_token: (self.special_token_freq, 0),
            self.pad_token: (self.special_token_freq, 1),
            self.start_token: (self.special_token_freq, 2),
            self.end_token: (self.special_token_freq, 3)
        }

    def map_words_to_freq_idx(self) -> Dict[str, Tuple[int, int]]:
        """
        Build vocab from instances
        return the word -> (freq, idx)
        :return:
        """
        all_words = []
        for instance in self.instances:
            all_words.extend(instance)

        # counter will map a list to Dict[str, count] values
        counter = Counter(all_words)

        # order the order in decreasing order of their frequencies
        # List[Tuple]
        counter = sorted(counter.items(),key=itemgetter(1), reverse=True)

        vocab = {}

        for idx, (word, freq) in enumerate(counter):
            vocab[word] = (freq, len(self.special_vocab) + idx)

        # merge the two dictionaries
        # courtesy https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        vocab = {**vocab, **self.special_vocab}

        return vocab

    def clip_on_mincount(self,
                         vocab: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Clip the vocab based on min count
        We decide to keep the word and it count
        We just change the idx of the token to idx of the unknown token
        :return: vocab: type: Dict[str, Tuple[int, int]]
        The new vocab
        """
        for key, (freq, idx) in vocab.items():
            if freq < self.min_count:
                vocab[key] = (freq, vocab[self.unk_token][1])

        return vocab

    def clip_on_max_num(self,
                        vocab: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Clip the vocab based on the maximum number of words
        We return `max_num_words + len(self.special_vocab)` words effectively
        The rest of them will be mapped to `self.unk_token`
        :param vocab: type: Dict[str, Tuple[int, int]]
        :return: vocab: type: Dict[str, Tuple[int, int]]
        The new vocab
        """
        for key, (freq, idx) in vocab.items():
            if idx >= len(self.special_vocab) + self.max_num_words:
                vocab[key] = (freq, vocab[self.unk_token][1])

        return vocab

    def build_vocab(self) -> Dict[str, Tuple[int, int]]:

        if self.store_location and os.path.isfile(self.store_location):
            self.load_from_file(self.store_location)
            self.msg_printer.good('Loaded vocab from file {0}'.format(self.store_location))

        else:
            self.msg_printer.info("BUILDING VOCAB")
            vocab = self.map_words_to_freq_idx()
            self.orig_vocab = vocab
            vocab = self.clip_on_mincount(vocab)
            vocab = self.clip_on_max_num(vocab)
            self.vocab = vocab
            self.idx2token = self.get_idx2token_mapping()
            self.token2idx = self.get_token2idx_mapping()

            if self.store_location:
                self.msg_printer.info('SAVING VOCAB TO FILE')
                self.save_to_file(self.store_location)
            return vocab
        self.msg_printer.good('Finished vocab loading')

    def get_vocab_len(self) -> int:
        if not self.vocab:
            raise ValueError('Build vocab first by calling build_vocab()')

        length = len(set(idx for freq, idx in self.vocab.values()))
        return length

    def get_orig_vocab_len(self) -> int:
        if not self.orig_vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        length = len(set(idx for freq, idx in self.vocab.values()))
        return length

    def get_token2idx_mapping(self) -> Dict[str, int]:
        if not self.vocab:
            raise ValueError('Build vocab first by calling build_vocab()')

        token2idx = {}
        for word, (freq, idx) in self.vocab.items():
            token2idx[word] = idx

        return token2idx

    def get_idx2token_mapping(self) -> Dict[int, str]:
        if not self.vocab:
            raise ValueError('Build vocab first by calling build_vocab()')

        idx2words = {}
        for word, (freq, idx) in self.vocab.items():
            idx2words[idx] = word
        return idx2words

    def save_to_file(self,
                     filename: str):
        """
        :param filename: str
        The filename where the result to the file will be stored
        The vocab will be stored in the json file name
        Please make sure that this is a json filename

        :return: None
        The whole vocab object will be saved to the file
        """

        if not self.vocab:
            raise ValueError('Build vocab first by calling build_vocab()')

        vocab_state = dict()
        vocab_state['options'] = {
            'max_num_words': self.max_num_words,
            'min_count': self.min_count,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'special_token_freq': self.special_token_freq
        }
        vocab_state['vocab'] = self.vocab
        vocab_state['orig_vocab'] = self.orig_vocab
        try:
            with open(filename, 'w') as fp:
                json.dump(vocab_state, fp)

        except FileNotFoundError:
            print("You passed {0} for the filename. Please check whether "
                  "the path exists and try again".format(filename))

    def load_from_file(self, filename: str) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]:
        try:
            with open(filename, 'r') as fp:
                vocab_state = json.load(fp)
                vocab_options = vocab_state['options']
                vocab = vocab_state['vocab']
                orig_vocab = vocab_state['orig_vocab']

                # restore the object
                # restore all the property values from the file

                self.vocab = vocab
                self.orig_vocab = orig_vocab
                self.token2idx = self.get_token2idx_mapping()
                self.idx2token = self.get_idx2token_mapping()
                self.max_num_words = vocab_options['max_num_words']
                self.min_count = vocab_options['min_count']
                self.unk_token = vocab_options['unk_token']
                self.pad_token = vocab_options['pad_token']
                self.start_token = vocab_options['start_token']
                self.end_token = vocab_options['end_token']
                self.special_token_freq = vocab_options['special_token_freq']

                return vocab_options, vocab
        except FileNotFoundError:
            print("You passed {0} for the filename. Please check whether "
                  "the path exists and try again. Please pass "
                  "a json file".format(filename))

    def get_token_from_idx(self,
                           idx: int)-> str:
        if not self.vocab:
            raise ValueError("Please build the vocab first")

        if not self.idx2token:
            self.idx2token = self.get_idx2token_mapping()

        try:
            return self.idx2token[idx]
        except KeyError:
            vocab_len = self.get_vocab_len()
            message = "You tried to access idx {0} of the vocab " \
                      "The length of the vocab is {1}. Please Provide " \
                      "Number between {2}".format(idx, vocab_len, vocab_len-1)
            raise ValueError(message)

    def get_idx_from_token(self,
                           token: str) -> int:
        if not self.vocab:
            raise ValueError("Please build the vocab first")

        if not self.token2idx:
            self.token2idx = self.get_token2idx_mapping()

        try:
            return self.token2idx[token]
        except KeyError:
            return self.token2idx[self.unk_token]

    def get_topn_frequent_words(self,
                                n: int=5) -> List[Tuple[str, int]]:
        idx2token = self.idx2token
        token_freqs = []
        max_n = min(len(self.special_vocab) + n, self.get_vocab_len())
        for idx in range(len(self.special_vocab), max_n):
            token = idx2token[idx]
            freq = self.orig_vocab[token][0]
            token_freqs.append((token, freq))

        return token_freqs

    def print_stats(self) -> None:
        orig_vocab_len = self.get_orig_vocab_len()
        vocab_len = self.get_vocab_len()
        N = 5
        top_n = self.get_topn_frequent_words(n=N)

        data = [('Original vocab length', orig_vocab_len),
                ('Clipped vocab length', vocab_len),
                ('Top {0} words'.format(N), top_n)]
        header = ("-", "Stats")
        table_string = wasabi.table(
                     data=data,
                     header=header,
                     divider=True
        )
        print(table_string)


