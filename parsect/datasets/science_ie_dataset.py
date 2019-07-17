from typing import List, Dict, Any
from torch.utils.data import Dataset
from parsect.datasets.TextClassificationDataset import TextClassificationDataset
from typing import Union
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.tokenizers.character_tokenizer import CharacterTokenizer
import numpy as np
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
import wasabi
from parsect.utils.vis_seq_tags import VisTagging
from parsect.utils.common import pack_to_length
import torch
import collections


class ScienceIEDataset(TextClassificationDataset, Dataset):
    def __init__(
        self,
        science_ie_conll_file: str,
        dataset_type: str,
        max_num_words: int,
        max_word_length: int,
        max_char_length: int,
        word_vocab_store_location: str,
        char_vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[str, None] = None,
        character_embedding_dimension: Union[str, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenizer=WordTokenizer("vanilla"),
        word_tokenization_type="vanilla",
        word_add_start_end_token: bool = True,
        character_tokenizer=CharacterTokenizer(),
    ):
        super(ScienceIEDataset, self).__init__(
            filename=science_ie_conll_file,
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
            validation_size=validation_size,
            test_size=test_size,
            word_tokenizer=word_tokenizer,
            word_tokenization_type=word_tokenization_type,
            character_tokenizer=character_tokenizer,
        )
        self.char_vocab_store_location = char_vocab_store_location
        self.character_embedding_dimension = character_embedding_dimension
        self.max_char_length = max_char_length
        self.word_add_start_end_token = word_add_start_end_token
        self.entity_types = ["Task", "Process", "Material"]

        self.classnames2idx = self.get_classname2idx()
        self.idx2classnames = {}
        for entity_type in self.entity_types:
            classnames2idx = self.classnames2idx[entity_type]
            self.idx2classnames[entity_type] = {v: k for k, v in classnames2idx.items()}

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

        self.msg_printer = wasabi.Printer()
        self.tag_visualizer = VisTagging()

    @staticmethod
    def get_classname2idx() -> Dict[str, Any]:
        tag_types = ["O", "B", "I", "L", "U"]
        classnames = {
            "Task": [f"{each_type}-Task" for each_type in tag_types],
            "Process": [f"{each_type}-Process" for each_type in tag_types],
            "Material": [f"{each_type}-Material" for each_type in tag_types],
        }
        classnames2idx_mapping = {}
        for entity_type in ["Task", "Process", "Material"]:
            entity_type_classnames = classnames[entity_type]
            classnames2idx_mapping[entity_type] = dict(
                [
                    (class_name, idx)
                    for idx, class_name in enumerate(entity_type_classnames)
                ]
            )

        return classnames2idx_mapping

    def get_num_classes(self) -> int:
        return len(self.classnames2idx["Task"])

    def get_class_names_from_indices(self, indices: List[int]):
        class_names = {}
        for entity_type in self.entity_types:
            class_names[entity_type] = [
                self.idx2classnames[entity_type][idx] for idx in indices
            ]

        return class_names

    def get_disp_sentence_from_indices(self, indices: List[int]):
        tokens = [
            self.word_vocab.get_token_from_idx(idx)
            for idx in indices
            if idx != self.word_vocab.special_vocab[self.word_vocab.pad_token][1]
        ]
        sentence = " ".join(tokens)
        return sentence

    def get_stats(self):
        num_instances = len(self.word_instances)
        len_instances = [len(instance) for instance in self.word_instances]
        max_len_instance = max(len_instances)
        all_task_labels = []
        all_process_labels = []
        all_material_labels = []
        for idx in range(num_instances):
            iter_dict = self[idx]
            labels = iter_dict["label"]
            task_labels = labels[0]
            process_labels = labels[1]
            material_labels = labels[2]
            all_task_labels.extend(task_labels.cpu().tolist())
            all_process_labels.extend(process_labels.cpu().tolist())
            all_material_labels.extend(material_labels.cpu().tolist())

        all_labels = {
            "Task": all_task_labels,
            "Material": all_material_labels,
            "Process": all_process_labels,
        }

        for entity_type in self.entity_types:
            task_labels_stats = dict(collections.Counter(all_task_labels))
            classes = list(set(task_labels_stats.keys()))
            classes = sorted(classes)
            header = ["label index", "label name", "count"]
            rows = [
                (
                    class_,
                    self.idx2classnames[entity_type][class_],
                    task_labels_stats[class_],
                )
                for class_ in classes
            ]
            formatted = wasabi.table(data=rows, header=header, divider=True)
            self.msg_printer.divider(
                f"Label Stats for Science IE {self.dataset_type} dataset with Entity Type {entity_type}"
            )
            print(formatted)

        # print some other stats
        random_int = np.random.randint(0, num_instances, size=1)[0]
        random_instance = self.word_instances[random_int]
        random_label = self.labels[random_int].split()
        random_task_label = [label.split(":")[0] for label in random_label]
        random_process_label = [label.split(":")[1] for label in random_label]
        random_material_label = [label.split(":")[2] for label in random_label]
        assert len(random_instance) == len(random_label)
        self.msg_printer.divider(
            f"Random Instance from Parscit {self.dataset_type.capitalize()} Dataset"
        )
        self.msg_printer.text(title="Task Labels")
        tagged_string = self.tag_visualizer.visualize_tokens(
            random_instance, random_task_label
        )
        print(tagged_string)

        self.msg_printer.text(title="Process Labels")
        tagged_string = self.tag_visualizer.visualize_tokens(
            random_instance, random_process_label
        )
        print(tagged_string)

        self.msg_printer.text(title="Material Labels")
        tagged_string = self.tag_visualizer.visualize_tokens(
            random_instance, random_material_label
        )
        print(tagged_string)

        num_instances = len(self)
        other_stats_header = ["", "Value"]
        rows = [
            ("Num Instances", num_instances),
            ("Longest Instance Length", max_len_instance),
        ]

        other_stats_table = wasabi.table(
            data=rows, header=other_stats_header, divider=True
        )
        self.msg_printer.divider(
            f"Other stats for ScienceIE {self.dataset_type} dataset"
        )
        print(other_stats_table)

    def get_lines_labels(self):
        lines = []
        labels = []  # contains three labels per word

        with open(self.filename) as fp:
            words = []
            tags = []
            for line in fp:
                if bool(line.strip()):
                    word, task_tag, process_tag, material_tag = line.strip().split()
                    words.append(word)
                    tags.append(":".join([task_tag, process_tag, material_tag]))
                else:
                    citation = " ".join(words)
                    label = " ".join(tags)

                    assert len(words) == len(tags)
                    labels.append(label)
                    lines.append(citation)
                    words = []
                    tags = []

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
        self.word_vocab.load_embedding()

    def get_preloaded_char_embedding(self):
        return self.char_vocab.load_embedding(embedding_for="character")

    def __len__(self):
        return len(self.word_instances)

    def __getitem__(self, idx):
        word_instance = self.word_instances[idx]
        labels_string = self.labels[idx]
        labels_string = labels_string.split()
        len_instance = len(word_instance)

        assert len_instance == len(labels_string)

        task_labels = []
        process_labels = []
        material_labels = []

        for string in labels_string:
            task_label, process_label, material_label = string.split(":")
            task_labels.append(task_label)
            process_labels.append(process_label)
            material_labels.append(material_label)

        assert len_instance == len(task_labels)
        assert len_instance == len(process_labels)
        assert len_instance == len(material_labels)

        padded_word_instance = pack_to_length(
            tokenized_text=word_instance,
            max_length=self.max_length,
            pad_token=self.word_vocab.pad_token,
            add_start_end_token=self.word_add_start_end_token,
            start_token=self.word_vocab.start_token,
            end_token=self.word_vocab.end_token,
        )

        padded_task_labels = pack_to_length(
            tokenized_text=task_labels,
            max_length=self.max_length,
            pad_token="padding",
            add_start_end_token=self.word_add_start_end_token,
            start_token="starting",
            end_token="ending",
        )

        padded_process_labels = pack_to_length(
            tokenized_text=process_labels,
            max_length=self.max_length,
            pad_token="padding",
            add_start_end_token=self.word_add_start_end_token,
            start_token="starting",
            end_token="ending",
        )

        padded_material_labels = pack_to_length(
            tokenized_text=material_labels,
            max_length=self.max_length,
            pad_token="padding",
            add_start_end_token=self.word_add_start_end_token,
            start_token="starting",
            end_token="ending",
        )

        assert len(padded_word_instance) == len(padded_task_labels)
        assert len(padded_word_instance) == len(padded_process_labels)
        assert len(padded_word_instance) == len(padded_material_labels)

        tokens = self.word_numericalizer.numericalize_instance(padded_word_instance)
        padded_task_labels = [
            self.classnames2idx["Task"][label] for label in padded_task_labels
        ]
        padded_process_labels = [
            self.classnames2idx["Process"][label] for label in padded_process_labels
        ]
        padded_material_labels = [
            self.classnames2idx["Material"][label] for label in padded_material_labels
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
        task_label = torch.LongTensor(padded_task_labels)
        process_label = torch.LongTensor(padded_process_labels)
        material_label = torch.LongTensor(padded_material_labels)
        character_tokens = torch.LongTensor(character_tokens)
        label = torch.stack([task_label, process_label, material_label], dim=0)

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "instance": " ".join(padded_word_instance),
            "raw_instance": " ".join(word_instance),
            "char_tokens": character_tokens,
            "label": label,
        }
        return instance_dict


if __name__ == "__main__":
    import pathlib
    import parsect.constants as constants

    PATHS = constants.PATHS
    DATA_DIR = PATHS["DATA_DIR"]
    data_dir_path = pathlib.Path(DATA_DIR)
    train_science_conll_file = data_dir_path.joinpath("train_science_conll.txt")
    vocab_store_location = data_dir_path.joinpath("vocab.json")
    char_vocab_store_location = data_dir_path.joinpath("char_vocab.json")
    DEBUG = False
    MAX_NUM_WORDS = 10000
    MAX_LENGTH = 20
    MAX_CHAR_LENGTH = 25
    EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 25

    dataset = ScienceIEDataset(
        science_ie_conll_file=train_science_conll_file,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=str(vocab_store_location),
        debug=DEBUG,
        word_embedding_type="random",
        word_embedding_dimension=EMBEDDING_DIM,
        word_add_start_end_token=False,
        char_vocab_store_location=str(char_vocab_store_location),
        character_embedding_dimension=CHAR_EMBEDDING_DIM,
    )

    dataset.get_stats()
    vocab_store_location.unlink()
    char_vocab_store_location.unlink()
