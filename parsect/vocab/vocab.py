from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from operator import itemgetter
import json
import os
from wasabi import Printer
import wasabi
from copy import deepcopy
from parsect.vocab.word_emb_loader import WordEmbLoader
from parsect.vocab.char_emb_loader import CharEmbLoader
import torch
from typing import Union


class Vocab:
    def __init__(
        self,
        instances: Optional[List[List[str]]] = None,
        max_num_tokens: int = None,
        min_count: int = 1,
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        special_token_freq: float = 1e10,
        store_location: str = None,
        embedding_type: Union[str, None] = None,
        embedding_dimension: Union[int, None] = None,
    ):
        """

        :param instances: type: List[List[str]]
         Pass in the list of tokenized instances from which vocab is built
        :param max_num_tokens: type: int
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
        :param embedding_type: type: str
        The embedding type is the type of pre-trained embedding that will be loaded
        for all the words in the vocab optionally. You can refer to `WordEmbLoder`
        for all the available embedding types
        :param embedding_dimension: type: int
        Embedding dimension of the embedding type
        """
        self.instances = instances
        self.max_num_tokens = max_num_tokens
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
        self.embedding_type = embedding_type
        self.embedding_dimension = embedding_dimension

        self.msg_printer = Printer()

        # store the special tokens
        self.special_vocab = {
            self.unk_token: (self.special_token_freq, 0),
            self.pad_token: (self.special_token_freq, 1),
            self.start_token: (self.special_token_freq, 2),
            self.end_token: (self.special_token_freq, 3),
        }

    def map_tokens_to_freq_idx(self) -> Dict[str, Tuple[int, int]]:
        """
        Build vocab from instances
        return the word -> (freq, idx)
        :return:
        """
        all_tokens = []
        for instance in self.instances:
            all_tokens.extend(instance)

        # counter will map a list to Dict[str, count] values
        counter = Counter(all_tokens)

        # order the order in decreasing order of their frequencies
        # List[Tuple]
        counter = sorted(counter.items(), key=itemgetter(1), reverse=True)

        vocab = {}

        for idx, (token, freq) in enumerate(counter):
            vocab[token] = (freq, len(self.special_vocab) + idx)

        # merge the two dictionaries
        # courtesy https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        vocab = {**vocab, **self.special_vocab}

        # BUG: if vocab and special vocab share same token, then
        # the index of the vocab will get overwritten by special vocab
        # the only way now is to recalculate indices based on frequencies
        vocab = sorted(vocab.items(), key=itemgetter(1), reverse=True)
        new_vocab = {}
        for idx, (token, (freq, _)) in enumerate(vocab):
            new_vocab[token] = (freq, idx)
        return new_vocab

    def clip_on_mincount(
        self, vocab: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Tuple[int, int]]:
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

    def clip_on_max_num(
        self, vocab: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Clip the vocab based on the maximum number of words
        We return `max_num_words + len(self.special_vocab)` words effectively
        The rest of them will be mapped to `self.unk_token`
        :param vocab: type: Dict[str, Tuple[int, int]]
        :return: vocab: type: Dict[str, Tuple[int, int]]
        The new vocab
        """
        for key, (freq, idx) in vocab.items():
            if idx >= len(self.special_vocab) + self.max_num_tokens:
                vocab[key] = (freq, vocab[self.unk_token][1])

        return vocab

    def _add_token(self, token: str, save_vocab: bool = False):
        """
        Add token to an already existing vocabulary
        :param token: type str
        :return:
        """
        try:
            vocab = self.vocab
        except AttributeError:
            self.msg_printer.fail("Please build vocab using build vocab")
        tokens = vocab.keys()
        indices = [idx for freq, idx in vocab.values()]
        indices = sorted(indices, reverse=True)
        highest_idx = indices[0]

        if token not in tokens:
            self.vocab[token] = (1, highest_idx + 1)
            self.idx2token[highest_idx + 1] = token
            self.token2idx[token] = highest_idx + 1
            if save_vocab:
                self.save_to_file(self.store_location)  # this can be expensive.

    def add_tokens(self, tokens: List[str]):
        try:
            vocab = self.vocab
        except AttributeError:
            self.msg_printer.fail("Please build vocab first")

        for token in tokens:
            self._add_token(token, save_vocab=False)

        if self.store_location:
            self.save_to_file(self.store_location)

    def build_vocab(self) -> Dict[str, Tuple[int, int]]:

        if self.store_location and os.path.isfile(self.store_location):
            vocab_object = self.load_from_file(self.store_location)
            self.msg_printer.good(
                "Loaded vocab from file {0}".format(self.store_location)
            )
            self.vocab = vocab_object.vocab
            self.orig_vocab = vocab_object.orig_vocab
            self.idx2token = vocab_object.idx2token
            self.token2idx = vocab_object.token2idx
            vocab = vocab_object.vocab

        else:
            self.msg_printer.info("BUILDING VOCAB")
            vocab = self.map_tokens_to_freq_idx()
            self.orig_vocab = deepcopy(
                vocab
            )  # dictionary are passed by reference. Be careful
            vocab = self.clip_on_mincount(vocab)
            vocab = self.clip_on_max_num(vocab)
            self.vocab = vocab
            self.idx2token = self.get_idx2token_mapping()
            self.token2idx = self.get_token2idx_mapping()

            if self.store_location:
                self.msg_printer.info("SAVING VOCAB TO FILE")
                self.save_to_file(self.store_location)
        return vocab

    def get_vocab_len(self) -> int:
        if not self.vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        length = len(set(idx for freq, idx in self.vocab.values()))
        return length

    def get_orig_vocab_len(self) -> int:
        if not self.orig_vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        length = len(set(idx for freq, idx in self.orig_vocab.values()))
        return length

    def get_token2idx_mapping(self) -> Dict[str, int]:
        if not self.vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        token2idx = {}
        for word, (freq, idx) in self.vocab.items():
            token2idx[word] = idx

        return token2idx

    def get_idx2token_mapping(self) -> Dict[int, str]:
        if not self.vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        idx2words = {}
        for word, (freq, idx) in self.vocab.items():
            idx2words[idx] = word
        return idx2words

    def save_to_file(self, filename: str):
        """
        :param filename: str
        The filename where the result to the file will be stored
        The vocab will be stored in the json file name
        Please make sure that this is a json filename

        :return: None
        The whole vocab object will be saved to the file
        """

        if not self.vocab:
            raise ValueError("Build vocab first by calling build_vocab()")

        vocab_state = dict()
        vocab_state["options"] = {
            "max_num_words": self.max_num_tokens,
            "min_count": self.min_count,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "special_token_freq": self.special_token_freq,
            "embedding_type": self.embedding_type,
            "embedding_dimension": self.embedding_dimension,
            "special_vocab": self.special_vocab,
        }
        vocab_state["vocab"] = self.vocab
        vocab_state["orig_vocab"] = self.orig_vocab
        try:
            with open(filename, "w") as fp:
                json.dump(vocab_state, fp)

        except FileNotFoundError:
            print(
                "You passed {0} for the filename. Please check whether "
                "the path exists and try again".format(filename)
            )

    @classmethod
    def load_from_file(cls, filename: str) -> "Vocab":
        try:
            with open(filename, "r") as fp:
                vocab_state = json.load(fp)
                vocab_options = vocab_state["options"]
                vocab_dict = vocab_state["vocab"]
                orig_vocab_dict = vocab_state["orig_vocab"]

                # restore the object
                # restore all the property values from the file

                max_num_tokens = vocab_options["max_num_words"]
                min_count = vocab_options["min_count"]
                unk_token = vocab_options["unk_token"]
                pad_token = vocab_options["pad_token"]
                start_token = vocab_options["start_token"]
                end_token = vocab_options["end_token"]
                special_token_freq = vocab_options["special_token_freq"]
                store_location = filename
                embedding_type = vocab_options.get("embedding_type")
                embedding_dimension = vocab_options.get("embedding_dimension")
                vocab = cls(
                    max_num_tokens=max_num_tokens,
                    min_count=min_count,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    start_token=start_token,
                    end_token=end_token,
                    instances=None,
                    special_token_freq=special_token_freq,
                    store_location=store_location,
                    embedding_type=embedding_type,
                    embedding_dimension=embedding_dimension,
                )

                # instead of building the vocab, set the vocab from vocab_dict
                vocab.set_vocab(vocab=vocab_dict)
                vocab.set_orig_vocab(orig_vocab_dict)
                idx2token = vocab.get_idx2token_mapping()
                token2idx = vocab.get_token2idx_mapping()
                vocab.set_idx2token(idx2token)
                vocab.set_token2idx(token2idx)

                return vocab
        except FileNotFoundError:
            print(
                "You passed {0} for the filename. Please check whether "
                "the path exists and try again. Please pass "
                "a json file".format(filename)
            )

    def get_token_from_idx(self, idx: int) -> str:
        if not self.vocab:
            raise ValueError("Please build the vocab first")

        if not self.idx2token:
            self.idx2token = self.get_idx2token_mapping()

        try:
            return self.idx2token[idx]
        except KeyError:
            vocab_len = self.get_vocab_len()
            message = (
                "You tried to access idx {0} of the vocab "
                "The length of the vocab is {1}. Please Provide "
                "Number between {2}".format(idx, vocab_len, vocab_len - 1)
            )
            raise ValueError(message)

    def get_idx_from_token(self, token: str) -> int:
        if not self.vocab:
            raise ValueError("Please build the vocab first")

        if not self.token2idx:
            self.token2idx = self.get_token2idx_mapping()

        try:
            return self.token2idx[token]
        except KeyError:
            return self.token2idx[self.unk_token]

    def get_topn_frequent_words(self, n: int = 5) -> List[Tuple[str, int]]:
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

        data = [
            ("Original vocab length", orig_vocab_len),
            ("Clipped vocab length", vocab_len),
            ("Top {0} words".format(N), top_n),
        ]
        header = ("Stats Description", "#")
        table_string = wasabi.table(data=data, header=header, divider=True)
        self.msg_printer.divider("VOCAB STATS")
        print(table_string)

    def load_embedding(self, embedding_for: str = "word") -> torch.FloatTensor:
        if not self.vocab:
            raise ValueError("Please build the vocab first")

        embedding_loader = None
        if embedding_for == "word":
            embedding_loader = WordEmbLoader(
                token2idx=self.token2idx,
                embedding_type=self.embedding_type,
                embedding_dimension=self.embedding_dimension,
            )
        elif embedding_for == "character":
            embedding_loader = CharEmbLoader(
                token2idx=self.token2idx,
                embedding_type="random",
                embedding_dimension=self.embedding_dimension,
            )

        indices = [key for key in self.idx2token.keys()]
        indices = sorted(indices)

        embeddings = []
        for idx in indices:
            token = self.idx2token[idx]
            # numpy array appends to the embeddings array
            embedding = embedding_loader.vocab_embedding[token]
            embeddings.append(embedding)

        embeddings = torch.FloatTensor(embeddings)
        return embeddings

    def set_vocab(self, vocab: Dict[str, Tuple[int, int]]):
        self.vocab = vocab

    def set_orig_vocab(self, orig_vocab: Dict[str, Tuple[int, int]]):
        self.orig_vocab = orig_vocab

    def set_idx2token(self, idx2token: Dict[int, str]):
        self.idx2token = idx2token

    def set_token2idx(self, token2idx: Dict[str, int]):
        self.token2idx = token2idx
