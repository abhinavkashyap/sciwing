from typing import List, Dict, Union, Optional
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.tokenizers.character_tokenizer import CharacterTokenizer
from parsect.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
from parsect.utils.vis_seq_tags import VisTagging
import wasabi
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.utils.common import pack_to_length
import numpy as np
import torch
from parsect.preprocessing.instance_preprocessing import InstancePreprocessing
from parsect.datasets.sprinkle_dataset import sprinkle_dataset
from parsect.utils.class_nursery import ClassNursery


@sprinkle_dataset(vocab_pipe=["word_vocab", "char_vocab"])
class ParscitDataset(BaseSeqLabelingDataset, ClassNursery):
    ignored_labels = ["padding", "starting", "ending"]

    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_instance_length: int,
        word_vocab_store_location: str,
        max_char_length: Optional[int] = None,
        char_vocab_store_location: Optional[str] = None,
        captialization_vocab_store_location: Optional[str] = None,
        capitalization_emb_dim: Optional[str] = None,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[int, None] = None,
        char_embedding_dimension: Union[int, None] = None,
        word_start_token: str = "<SOS>",
        word_end_token: str = "<EOS>",
        word_pad_token: str = "<PAD>",
        word_unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenization_type="vanilla",
        word_add_start_end_token: bool = True,
        max_num_chars: Optional[int] = 10000,
        char_embedding_type: str = "random",
        char_unk_token: str = " ",
        char_pad_token: str = " ",
        char_end_token: str = " ",
        char_start_token: str = " ",
    ):

        self.filename = filename
        self.train_size = train_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.dataset_type = dataset_type
        self.debug = debug
        self.debug_dataset_proportion = debug_dataset_proportion
        self.max_instance_length = max_instance_length

        self.word_add_start_end_token = word_add_start_end_token
        self.classnames2idx = self.get_classname2idx()
        self.instance_preprocessor = None
        self.idx2classname = {
            idx: classname for classname, idx in self.classnames2idx.items()
        }

        self.lines, self.labels = self.get_lines_labels(filename)

        self.msg_printer = wasabi.Printer()
        self.tag_visualizer = VisTagging()

    @classmethod
    def get_classname2idx(cls) -> Dict[str, int]:
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
        categories["notes"] = categories["note"]
        return categories

    def get_num_classes(self) -> int:
        return len(set(self.classnames2idx.values()))

    def get_class_names_from_indices(self, indices: List[int]):
        return [self.idx2classname[idx] for idx in indices]

    def print_stats(self):
        num_instances = self.num_instances
        formatted = self.label_stats_table
        self.msg_printer.divider(f"Label Stats for Parscit {self.dataset_type} dataset")
        print(formatted)

        # print some other stats
        random_int = np.random.randint(0, num_instances, size=1)[0]
        random_instance = self.word_instances[random_int]
        random_label = self.labels[random_int].split()
        assert len(random_instance) == len(random_label)
        self.msg_printer.divider(
            f"Random Instance from Parscit {self.dataset_type.capitalize()} Dataset"
        )
        tagged_string = self.tag_visualizer.visualize_tokens(
            random_instance, random_label
        )
        print(tagged_string)

        num_instances = len(self)
        other_stats_header = ["", "Value"]
        rows = [
            ("Num Instances", num_instances),
            ("Longest Instance Length", self.instance_max_len),
        ]

        other_stats_table = wasabi.table(
            data=rows, header=other_stats_header, divider=True
        )
        self.msg_printer.divider(f"Other stats for Parscit {self.dataset_type} dataset")
        print(other_stats_table)

    def get_lines_labels(self, filename: str) -> (List[str], List[str]):
        """
        Returns citation string, label
        citation string is a space separated string
        label is a space separated string of different labels
        :return:
        """
        lines = []
        labels = []
        with open(filename) as fp:
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

    def __len__(self):
        return len(self.word_instances)

    def __getitem__(self, idx):
        line = self.lines[idx]
        labels = self.labels[idx].split()

        return self.get_iter_dict(line=line, label=labels)

    @classmethod
    def emits_keys(cls) -> Dict[str, str]:
        return {
            "tokens": f"A torch.LongTensor of size `max_word_length`. "
            f"Example [0, 0, 1, 100] where every number represents an index in the vocab and the "
            f"length is `max_word_length`",
            "len_tokens": f"A torch.LongTensor. "
            f"Example [2] representing the number of tokens without padding",
            "label": f"A torch.LongTensor representing the label corresponding to the "
            f"instance. Example [2,  0, 1] representing class [2, 0, 1] for an instance with "
            f"3 tokens. Here the length of the instance will match the length of the labels",
            "instance": f"A string that is padded to ``max_word_length``.",
            "raw_instance": f"A string that is not padded",
            "char_tokens": f"Every word is tokenized into characters of length `max_char_len`. "
            f"char_tokens for an instance is a torch.LongTensor of shape [max_word_len, max_char_len]",
        }

    def get_iter_dict(self, line: str, label: Optional[List[str]] = None):
        word_instance = self.word_tokenizer.tokenize(line)
        len_instance = len(word_instance)
        classnames2idx = ParscitDataset.get_classname2idx()
        idx2classname = {idx: classname for classname, idx in classnames2idx.items()}

        if self.instance_preprocessor is not None:
            word_instance = self.instance_preprocessor.lowercase(word_instance)

        padded_word_instance = pack_to_length(
            tokenized_text=word_instance,
            max_length=self.max_instance_length,
            pad_token=self.word_vocab.pad_token,
            add_start_end_token=self.word_add_start_end_token,
            start_token=self.word_vocab.start_token,
            end_token=self.word_vocab.end_token,
        )
        tokens = self.word_numericalizer.numericalize_instance(padded_word_instance)
        tokens = torch.LongTensor(tokens)
        len_tokens = torch.LongTensor([len_instance])

        character_tokens = []
        # 1. For every word we get characters in the word
        # 2. Pad the characters to max_char_length
        # 3. Convert them into numbers
        # 4. Add them to character_tokens
        for word in padded_word_instance:
            char_instance = self.char_tokenizer.tokenize(word)
            padded_character_instance = pack_to_length(
                tokenized_text=char_instance,
                max_length=self.max_char_length,
                pad_token=" ",
                add_start_end_token=False,
            )
            padded_character_tokens = self.char_numericalizer.numericalize_instance(
                padded_character_instance
            )
            character_tokens.append(padded_character_tokens)
        character_tokens = torch.LongTensor(character_tokens)

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "instance": " ".join(padded_word_instance),
            "raw_instance": " ".join(word_instance),
            "char_tokens": character_tokens,
        }

        if label is not None:
            assert len_instance == len(label)
            padded_labels = pack_to_length(
                tokenized_text=label,
                max_length=self.max_instance_length,
                pad_token="padding",
                add_start_end_token=self.word_add_start_end_token,
                start_token="starting",
                end_token="ending",
            )
            padded_labels = [classnames2idx[label] for label in padded_labels]
            labels_mask = []
            for class_idx in padded_labels:
                if idx2classname[class_idx] in self.ignored_labels:
                    labels_mask.append(1)
                else:
                    labels_mask.append(0)
            label = torch.LongTensor(padded_labels)
            labels_mask = torch.ByteTensor(labels_mask)

            instance_dict["label"] = label
            instance_dict["label_mask"] = labels_mask

        return instance_dict
