import pytest
from parsect.models.science_ie_tagger import ScienceIETagger
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.charlstm_encoder import CharLSTMEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.embedders.concat_embedders import ConcatEmbedders
from allennlp.modules.conditional_random_field import allowed_transitions
from parsect.datasets.seq_labeling.science_ie_dataset import ScienceIEDataset
import itertools
import torch.nn as nn
import torch
import numpy as np
from typing import List

lstm2encoder_options = itertools.product(
    [True, False], ["sum", "concat"], [True, False]
)
lstm2encoder_options = list(lstm2encoder_options)


@pytest.fixture(params=lstm2encoder_options)
def setup_science_ie_tagger(request):
    EMBEDDING_DIM = 100
    CHARACTER_EMBEDDING_DIM = 25
    VOCAB_SIZE = 1000
    BATCH_SIZE = 2
    HIDDEN_DIM = 1024
    CHARACTER_ENCODER_HIDDEN_DIM = 100
    NUM_TIME_STEPS = 10
    MAX_CHAR_LENGTH = 25
    CHAR_VOCAB_SIZE = 100
    BIDIRECTIONAL = request.param[0]
    COMBINE_STRATEGY = request.param[1]
    HAVE_CHARACTER_ENCODER = request.param[2]
    NUM_CLASSES = 8
    EMBEDDING = nn.Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMBEDDING_DIM]))
    CHARACTER_EMBEDDING = nn.Embedding.from_pretrained(
        torch.zeros([CHAR_VOCAB_SIZE, CHARACTER_EMBEDDING_DIM])
    )
    tokens = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, NUM_TIME_STEPS))

    task_labels = np.random.randint(0, 8, size=(BATCH_SIZE, NUM_TIME_STEPS))
    process_labels = np.random.randint(8, 16, size=(BATCH_SIZE, NUM_TIME_STEPS))
    material_labels = np.random.randint(16, 24, size=(BATCH_SIZE, NUM_TIME_STEPS))
    task_labels = torch.LongTensor(task_labels)
    process_labels = torch.LongTensor(process_labels)
    material_labels = torch.LongTensor(material_labels)
    labels = torch.cat([task_labels, process_labels, material_labels], dim=1)

    char_tokens = np.random.randint(
        0, CHAR_VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TIME_STEPS, MAX_CHAR_LENGTH)
    )
    tokens = torch.LongTensor(tokens)
    labels = torch.LongTensor(labels)
    char_tokens = torch.LongTensor(char_tokens)

    classnames2idx = ScienceIEDataset.get_classname2idx()
    idx2classnames = {idx: classname for classname, idx in classnames2idx.items()}
    task_idx2classnames = {
        idx: classname
        for idx, classname in idx2classnames.items()
        if idx in range(0, 8)
    }
    process_idx2classnames = {
        idx - 8: classname
        for idx, classname in idx2classnames.items()
        if idx in range(8, 16)
    }
    material_idx2classnames = {
        idx - 16: classname
        for idx, classname in idx2classnames.items()
        if idx in range(16, 24)
    }

    task_constraints: List[(int, int)] = allowed_transitions(
        constraint_type="BIOUL", labels=task_idx2classnames
    )
    process_constraints: List[(int, int)] = allowed_transitions(
        constraint_type="BIOUL", labels=process_idx2classnames
    )
    material_constraints: List[(int, int)] = allowed_transitions(
        constraint_type="BIOUL", labels=material_idx2classnames
    )

    embedder = VanillaEmbedder(embedding=EMBEDDING, embedding_dim=EMBEDDING_DIM)

    if HAVE_CHARACTER_ENCODER:
        char_embedder = VanillaEmbedder(
            embedding=CHARACTER_EMBEDDING, embedding_dim=CHARACTER_EMBEDDING_DIM
        )
        char_encoder = CharLSTMEncoder(
            char_embedder=char_embedder,
            char_emb_dim=CHARACTER_EMBEDDING_DIM,
            hidden_dim=CHARACTER_ENCODER_HIDDEN_DIM,
            bidirectional=True,
            combine_strategy="concat",
        )
        embedder = ConcatEmbedders([embedder, char_encoder])
        EMBEDDING_DIM += 2 * CHARACTER_ENCODER_HIDDEN_DIM

    encoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIM,
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
    )

    tagger = ScienceIETagger(
        rnn2seqencoder=encoder,
        hid_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        task_constraints=task_constraints,
        process_constraints=process_constraints,
        material_constraints=material_constraints,
    )

    return (
        tagger,
        {
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "VOCAB_SIZE": VOCAB_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "HIDDEN_DIM": 2 * HIDDEN_DIM
            if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
            else HIDDEN_DIM,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "tokens": tokens,
            "labels": labels,
            "EXPECTED_HIDDEN_DIM": 2 * HIDDEN_DIM
            if COMBINE_STRATEGY == "concat" and BIDIRECTIONAL
            else HIDDEN_DIM,
            "TIME_STEPS": NUM_TIME_STEPS,
            "NUM_CLASSES": NUM_CLASSES,
            "HAVE_CHAR_ENCODER": HAVE_CHARACTER_ENCODER,
            "char_tokens": char_tokens,
        },
    )


class TestScienceIETagger:
    def test_scienceie_tagger_dimensions(self, setup_science_ie_tagger):
        tagger, options = setup_science_ie_tagger
        tokens = options["tokens"]
        labels = options["labels"]
        char_tokens = options["char_tokens"]
        BATCH_SIZE = options["BATCH_SIZE"]
        TIME_STEPS = options["TIME_STEPS"]
        NUM_CLASSES = options["NUM_CLASSES"]

        iter_dict = {"tokens": tokens, "label": labels, "char_tokens": char_tokens}

        output_dict = tagger(
            iter_dict=iter_dict, is_training=True, is_validation=False, is_test=False
        )

        assert output_dict["task_logits"].size() == (
            BATCH_SIZE,
            TIME_STEPS,
            NUM_CLASSES,
        )
        assert output_dict["process_logits"].size() == (
            BATCH_SIZE,
            TIME_STEPS,
            NUM_CLASSES,
        )
        assert output_dict["material_logits"].size() == (
            BATCH_SIZE,
            TIME_STEPS,
            NUM_CLASSES,
        )
