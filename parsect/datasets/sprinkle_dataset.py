from typing import List, Optional
import collections
import inspect
from parsect.datasets.classification.base_text_classification import (
    BaseTextClassification,
)
from parsect.numericalizer.numericalizer import Numericalizer
from parsect.vocab.vocab import Vocab
import copy
import wrapt
import wasabi


class sprinkle_dataset:
    def __init__(self, vocab_pipe=None, autoset_attrs=True, get_label_stats_table=True):
        if vocab_pipe is None:
            vocab_pipe = ["word_vocab"]
        self.autoset_attrs = autoset_attrs
        self.vocab_pipe = vocab_pipe
        self.is_get_label_stats_table = get_label_stats_table
        self.wrapped_cls = None
        self.init_signature = None
        self.filename = None

        self.word_tokenizer = None
        self.word_instances = None
        self.word_vocab = None
        self.max_num_words = None
        self.word_vocab_store_location = None
        self.word_embedding_type = None
        self.word_embedding_dimension = None
        self.word_numericalizer = None
        self.word_unk_token = None
        self.word_pad_token = None
        self.word_start_token = None
        self.word_end_token = None

        self.char_tokenizer = None
        self.char_instances = None
        self.char_vocab = None
        self.max_num_chars = None
        self.char_vocab_store_location = None
        self.char_embedding_type = None
        self.char_embedding_dimension = None
        self.char_numericalizer = None
        self.char_unk_token = None
        self.char_pad_token = None
        self.char_start_token = None
        self.char_end_token = None

        self.word_vocab_required_attributes = [
            "word_tokenizer",
            "max_num_words",
            "word_vocab_store_location",
            "word_embedding_type",
            "word_embedding_type",
            "word_unk_token",
            "word_pad_token",
            "word_start_token",
            "word_end_token",
        ]

    def set_word_vocab(self):
        if not all(
            [
                attribute in dir(self)
                for attribute in self.word_vocab_required_attributes
            ]
        ):
            raise ValueError(
                f"For building word vocab, "
                f"please pass these attributes in your "
                f"dataset construction {self.word_vocab_required_attributes}"
            )
        self.word_instances = self.word_tokenizer.tokenize_batch(self.lines)
        self.word_vocab = Vocab(
            instances=self.word_instances,
            max_num_tokens=self.max_num_words,
            unk_token=self.word_unk_token,
            pad_token=self.word_pad_token,
            start_token=self.word_start_token,
            end_token=self.word_end_token,
            store_location=self.word_vocab_store_location,
            embedding_type=self.word_embedding_type,
            embedding_dimension=self.word_embedding_dimension,
        )
        self.word_numericalizer = Numericalizer(self.word_vocab)
        self.word_vocab.build_vocab()
        self.word_vocab.print_stats()

    def set_char_vocab(self):
        self.char_instances = self.char_tokenizer.tokenize_batch(self.lines)

        self.char_vocab = Vocab(
            instances=self.char_instances,
            max_num_tokens=1e6,
            min_count=1,
            store_location=self.char_vocab_store_location,
            embedding_type=self.char_embedding_type,
            embedding_dimension=self.char_embedding_dimension,
            start_token=self.char_start_token,
            end_token=self.char_end_token,
            unk_token=self.char_unk_token,
            pad_token=self.char_pad_token,
        )
        self.char_vocab.build_vocab()

        # adding these to help conversion to characters later
        self.char_vocab.add_tokens(
            list(self.word_start_token)
            + list(self.word_end_token)
            + list(self.word_unk_token)
            + list(self.word_pad_token)
        )
        self.char_numericalizer = Numericalizer(vocabulary=self.char_vocab)
        self.char_vocab.print_stats()

    def _get_label_stats_table(self):
        all_labels = []
        for label in self.labels:
            all_labels.extend(label.split())

        labels_stats = dict(collections.Counter(all_labels))
        classes = list(set(labels_stats.keys()))
        classes = sorted(classes)
        header = ["label index", "label name", "count"]
        classname2idx = self.wrapped_cls.get_classname2idx()
        rows = [
            (classname2idx[class_], class_, labels_stats[class_]) for class_ in classes
        ]
        formatted = wasabi.table(data=rows, header=header, divider=True)
        return formatted

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        self.wrapped_cls = wrapped
        self.init_signature = inspect.signature(wrapped.__init__)
        instance = wrapped(*args, **kwargs)
        for idx, (name, param) in enumerate(self.init_signature.parameters.items()):
            if name == "self":
                continue

            # These are values that must be passed
            if name in [
                "filename",
                "dataset_type",
                "max_num_words",
                "max_instance_length",
                "word_vocab_store_location",
            ]:
                try:
                    value = args[idx]
                except IndexError:
                    try:
                        value = kwargs[name]
                    except KeyError:
                        raise ValueError(
                            f"Dataset {self.cls.__name__} should be instantiated with {name}"
                        )
                if self.autoset_attrs:
                    setattr(instance, name, value)
                setattr(self, name, value)

            # These can be passed but have default values
            else:
                try:
                    value = args[idx]
                except IndexError:
                    try:
                        value = kwargs[name]
                    except KeyError:
                        value = param.default

                if self.autoset_attrs:
                    setattr(instance, name, value)
                setattr(self, name, value)

        # set the lines and labels
        self.lines, self.labels = instance.get_lines_labels(self.filename)
        self.word_instances = None
        self.word_vocab = None

        if "word_vocab" in self.vocab_pipe:
            self.set_word_vocab()
            instance.word_vocab = copy.deepcopy(self.word_vocab)
            instance.word_instances = copy.deepcopy(self.word_instances)
            instance.num_instances = len(self.word_instances)
            instance.instance_max_len = max(
                [len(instance) for instance in self.word_instances]
            )

        if "char_vocab" in self.vocab_pipe:
            self.set_char_vocab()
            instance.char_vocab = copy.deepcopy(self.char_vocab)
            instance.char_instances = copy.deepcopy(self.char_instances)

        if self.is_get_label_stats_table:
            label_stats_table = self._get_label_stats_table()
            instance.label_stats_table = label_stats_table

        return instance
