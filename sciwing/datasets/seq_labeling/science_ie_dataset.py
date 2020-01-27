import torch
from typing import List, Dict, Any, Optional
from sciwing.datasets.seq_labeling.base_seq_labeling import BaseSeqLabelingDataset
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.vocab.vocab import Vocab
from sciwing.numericalizers.numericalizer import Numericalizer
from sciwing.utils.vis_seq_tags import VisTagging
from sciwing.utils.common import pack_to_length
from typing import Union
import numpy as np
import wasabi
import collections
from sciwing.datasets.sprinkle_dataset import sprinkle_dataset
from sciwing.utils.class_nursery import ClassNursery


@sprinkle_dataset(vocab_pipe=["word_vocab", "char_vocab"], get_label_stats_table=False)
class ScienceIEDataset(BaseSeqLabelingDataset, ClassNursery):
    ignore_labels = [
        "starting-Task",
        "ending-Task",
        "padding-Task",
        "starting-Process",
        "ending-Process",
        "padding-Process",
        "starting-Material",
        "ending-Material",
        "padding-Material",
    ]

    def __init__(
        self,
        filename: str,
        dataset_type: str,
        max_num_words: int,
        max_instance_length: int,
        word_vocab_store_location: str,
        max_char_length: Optional[int] = None,
        char_vocab_store_location: Optional[str] = None,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[str, None] = None,
        char_embedding_dimension: Union[str, None] = None,
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
        self.entity_types = ["Task", "Process", "Material"]
        self.classnames2idx = self.get_classname2idx()
        self.idx2classnames = {v: k for k, v in self.classnames2idx.items()}

        self.lines, self.labels = self.get_lines_labels(self.filename)

        self.msg_printer = wasabi.Printer()
        self.tag_visualizer = VisTagging()

    @classmethod
    def get_classname2idx(cls) -> Dict[str, Any]:
        tag_types = ["O", "B", "I", "L", "U"]
        entity_types = ["Task", "Process", "Material"]
        class_names = []
        for entity_type in entity_types:
            class_names.extend([f"{tag_type}-{entity_type}" for tag_type in tag_types])
            class_names.extend(
                [
                    f"starting-{entity_type}",
                    f"ending-{entity_type}",
                    f"padding-{entity_type}",
                ]
            )

        classnames2idx_mapping = {}

        for idx, label in enumerate(class_names):
            classnames2idx_mapping[label] = idx

        return classnames2idx_mapping

    def get_num_classes(self) -> int:
        return 8

    def get_class_names_from_indices(self, indices: List[int]):
        return [self.idx2classnames[idx] for idx in indices]

    def print_stats(self):
        num_instances = len(self.word_instances)
        len_instances = [len(instance) for instance in self.word_instances]
        max_len_instance = max(len_instances)
        index_max_instance = len_instances.index(max_len_instance)

        all_task_labels = []
        all_process_labels = []
        all_material_labels = []
        for idx in range(num_instances):
            iter_dict = self[idx]
            labels = iter_dict["label"]
            task_labels, process_labels, material_labels = torch.chunk(
                labels, chunks=3, dim=0
            )

            all_task_labels.extend(task_labels.cpu().tolist())
            all_process_labels.extend(process_labels.cpu().tolist())
            all_material_labels.extend(material_labels.cpu().tolist())

        all_labels = {
            "Task": all_task_labels,
            "Process": all_process_labels,
            "Material": all_material_labels,
        }

        for entity_type in self.entity_types:
            label_stats = dict(collections.Counter(all_labels[entity_type]))
            classes = list(set(label_stats.keys()))
            classes = sorted(classes)
            header = ["label index", "label name", "count"]
            rows = [
                (class_, self.idx2classnames[class_], label_stats[class_])
                for class_ in classes
            ]
            formatted = wasabi.table(data=rows, header=header, divider=True)
            self.msg_printer.divider(
                f"Label Stats for Science IE {self.dataset_type} dataset with Entity Type {entity_type}"
            )
            print(formatted)

        # print some other stats
        random_instance = self.word_instances[index_max_instance]
        random_label = self.labels[index_max_instance].split()
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

    def get_lines_labels(self, filename: str):
        lines = []
        labels = []  # contains three labels per word

        with open(filename) as fp:
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

    def __len__(self):
        return len(self.word_instances)

    def __getitem__(self, idx):
        line = self.lines[idx]
        labels_string = self.labels[idx].split()

        return self.get_iter_dict(line=line, label=labels_string)

    @classmethod
    def emits_keys(cls):
        return {
            "tokens": f"A torch.LongTensor of size `max_word_length`. "
            f"Example [0, 0, 1, 100] where every number represents an index in the vocab and the "
            f"length is `max_word_length`",
            "len_tokens": f"A torch.LongTensor. "
            f"Example [2] representing the number of tokens without padding",
            "label": f"A torch.LongTensor representing the label corresponding to the "
            f"instance. Example [2,  0, 1] representing class [2, 0, 1] for an instance with "
            f"tokens. Here the length of the instance will match the length of the labels",
            "instance": f"A string that is padded to ``max_word_length``.",
            "raw_instance": f"A string that is not padded",
            "char_tokens": f"Every word is tokenized into characters of length `max_char_len`. "
            f"char_tokens for an instance is a torch.LongTensor of shape [max_word_len, max_char_len]",
        }

    def get_iter_dict(self, line: str, label: Optional[List[str]] = None):
        word_instance = self.word_tokenizer.tokenize(line)
        len_instance = len(word_instance)
        classnames2idx = ScienceIEDataset.get_classname2idx()
        idx2classname = {idx: classname for classname, idx in classnames2idx.items()}

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
            task_labels = []
            process_labels = []
            material_labels = []

            for string in label:
                task_label, process_label, material_label = string.split(":")
                task_labels.append(task_label)
                process_labels.append(process_label)
                material_labels.append(material_label)

            assert len_instance == len(task_labels)
            assert len_instance == len(process_labels)
            assert len_instance == len(material_labels)

            padded_task_labels = pack_to_length(
                tokenized_text=task_labels,
                max_length=self.max_instance_length,
                pad_token="padding-Task",
                add_start_end_token=self.word_add_start_end_token,
                start_token="starting-Task",
                end_token="ending-Task",
            )

            padded_process_labels = pack_to_length(
                tokenized_text=process_labels,
                max_length=self.max_instance_length,
                pad_token="padding-Process",
                add_start_end_token=self.word_add_start_end_token,
                start_token="starting-Process",
                end_token="ending-Process",
            )

            padded_material_labels = pack_to_length(
                tokenized_text=material_labels,
                max_length=self.max_instance_length,
                pad_token="padding-Material",
                add_start_end_token=self.word_add_start_end_token,
                start_token="starting-Material",
                end_token="ending-Material",
            )
            assert (
                len(padded_task_labels)
                == len(padded_process_labels)
                == len(padded_material_labels)
            )
            padded_task_labels = [classnames2idx[label] for label in padded_task_labels]

            # Ugly offsetting because we are using continuous numbers for classes in all entity
            # types but science ie dataset requires 0
            padded_process_labels = [
                classnames2idx[label] for label in padded_process_labels
            ]
            padded_material_labels = [
                classnames2idx[label] for label in padded_material_labels
            ]

            mask_task_label = [
                1 if idx2classname[class_idx] in self.ignore_labels else 0
                for class_idx in padded_task_labels
            ]
            mask_process_label = [
                1 if idx2classname[class_idx] in self.ignore_labels else 0
                for class_idx in padded_process_labels
            ]
            mask_material_label = [
                1 if idx2classname[class_idx] in self.ignore_labels else 0
                for class_idx in padded_material_labels
            ]

            task_label = torch.LongTensor(padded_task_labels)
            process_label = torch.LongTensor(padded_process_labels)
            material_label = torch.LongTensor(padded_material_labels)
            mask_task_label = torch.ByteTensor(mask_task_label)
            mask_process_label = torch.ByteTensor(mask_process_label)
            mask_material_label = torch.ByteTensor(mask_material_label)

            label = torch.cat([task_label, process_label, material_label], dim=0)
            label_mask = torch.cat(
                [mask_task_label, mask_process_label, mask_material_label], dim=0
            )
            instance_dict["label"] = label
            instance_dict["label_mask"] = label_mask

        return instance_dict
