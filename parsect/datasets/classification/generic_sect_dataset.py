from torch.utils.data import Dataset
from typing import Union, Dict, List, Any, Optional
from parsect.utils.common import convert_generic_sect_to_json
import wasabi
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.utils.common import pack_to_length
import torch
import collections
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)


class GenericSectDataset(Dataset, BaseTextClassification):
    def __init__(
        self,
        generic_sect_filename: str,
        dataset_type: str,
        max_num_words: int,
        max_length: int,
        word_vocab_store_location: str,
        debug: bool = False,
        debug_dataset_proportion: float = 0.1,
        word_embedding_type: Union[str, None] = None,
        word_embedding_dimension: Union[int, None] = None,
        start_token: str = "<SOS>",
        end_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        train_size: float = 0.8,
        test_size: float = 0.2,
        validation_size: float = 0.5,
        word_tokenizer=WordTokenizer(),
        word_tokenization_type="vanilla",
        add_start_end_token: bool = True,
    ):
        super(GenericSectDataset, self).__init__(
            filename=generic_sect_filename,
            dataset_type=dataset_type,
            max_num_words=max_num_words,
            max_length=max_length,
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
        )
        self.msg_printer = wasabi.Printer()
        self.add_start_end_token = add_start_end_token

        self.label2idx = self.get_classname2idx()
        self.idx2label = {idx: class_name for class_name, idx in self.label2idx.items()}

        self.generic_sect_json = convert_generic_sect_to_json(self.filename)
        self.headers, self.labels = self.get_lines_labels()
        self.instances = self.word_tokenize(self.headers)

        self.vocab = Vocab(
            instances=self.instances,
            max_num_tokens=self.max_num_words,
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

        self.numericalizer = Numericalizer(self.vocab)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        line = self.headers[idx]
        label = self.labels[idx]

        return self.get_iter_dict(
            lines=line,
            word_vocab=self.vocab,
            word_tokenizer=self.word_tokenizer,
            max_word_length=self.max_length,
            word_add_start_end_token=True,
            labels=label,
        )

    def get_lines_labels(self) -> (List[str], List[str]):
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
        ) = self.get_train_valid_test_stratified_split(headers, labels, self.label2idx)

        if self.dataset_type == "train":
            return train_headers, train_labels
        elif self.dataset_type == "valid":
            return valid_headers, valid_labels
        elif self.dataset_type == "test":
            return test_headers, test_labels

    def get_preloaded_word_embedding(self) -> torch.FloatTensor:
        return self.vocab.load_embedding()

    @classmethod
    def get_classname2idx(cls) -> Dict[str, int]:
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

    def print_stats(self):
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

    @classmethod
    def emits_keys(cls) -> Dict[str, str]:
        return {
            "tokens": f"A torch.LongTensor of size `max_length`. "
            f"Example [0, 0, 1, 100] where every number represents an index in the vocab",
            "len_tokens": f"A torch.LongTensor. "
            f"Example [2] representing the number of tokens without padding",
            "label": f"A torch.LongTensor representing the label corresponding to the "
            f"instance. Example [2] representing class 2",
            "instance": f"A string that is padded to ``max_length``.",
            "raw_instance": f"A string that is not padded",
        }

    @classmethod
    def get_iter_dict(
        cls,
        lines: Union[List[str], str],
        word_vocab: Vocab,
        word_tokenizer: WordTokenizer,
        max_word_length: int,
        word_add_start_end_token: bool,
        labels: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(lines, str):
            lines = [lines]

        word_instances = word_tokenizer.tokenize_batch(lines)
        len_instances = [len(instance) for instance in word_instances]
        word_numericalizer = Numericalizer(vocabulary=word_vocab)
        classnames2idx = cls.get_classname2idx()

        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]

            labels = [classnames2idx[label] for label in labels]
            label = torch.LongTensor(labels)

        padded_instances = []
        for word_instance in word_instances:
            padded_instance = pack_to_length(
                tokenized_text=word_instance,
                max_length=max_word_length,
                pad_token=word_vocab.pad_token,
                add_start_end_token=True,  # TODO: remove hard coded value here
                start_token=word_vocab.start_token,
                end_token=word_vocab.end_token,
            )
            padded_instances.append(padded_instance)

        tokens = word_numericalizer.numericalize_batch_instances(padded_instances)
        tokens = torch.LongTensor(tokens)
        tokens = tokens.squeeze(0)

        instances = []
        for instance in padded_instances:
            instances.append(" ".join(instance))

        raw_instances = []
        for instance in word_instances:
            raw_instances.append(" ".join(instance))

        len_tokens = torch.LongTensor(len_instances)

        # squeeze the dimensions if there are more than one dimension

        if len(instances) == 1:
            instances = instances[0]
            raw_instances = raw_instances[0]

        instance_dict = {
            "tokens": tokens,
            "len_tokens": len_tokens,
            "instance": instances,
            "raw_instance": raw_instances,
        }

        if labels is not None:
            instance_dict["label"] = label

        return instance_dict
