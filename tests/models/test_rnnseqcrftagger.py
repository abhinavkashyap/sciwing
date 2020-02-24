import pytest
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)


@pytest.fixture(scope="session")
def seq_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train.txt")
    train_file.write(
        "word11_train###label1 word21_train###label2\nword12_train###label1 word22_train###label2 word32_train###label3"
    )

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev.txt")
    dev_file.write(
        "word11_dev###label1 word21_dev###label2\nword12_dev###label1 word22_dev###label2 word32_dev###label3"
    )

    test_file = tmpdir_factory.mktemp("test_data").join("test.txt")
    test_file.write(
        "word11_test###label1 word21_test###label2\nword12_test###label1 word22_test###label2 word32_test###label3"
    )

    data_manager = SeqLabellingDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return data_manager


@pytest.fixture()
def setup_parscit_tagger(seq_dataset_manager):
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "concat"
    dataset_manager = seq_dataset_manager

    embedder = WordEmbedder(embedding_type="glove_6B_50")

    char_embedder = CharEmbedder(
        char_embedding_dimension=10,
        hidden_dimension=20,
        datasets_manager=dataset_manager,
    )
    embedder = ConcatEmbedders([embedder, char_embedder])

    encoder = Lstm2SeqEncoder(
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
    )

    tagger = RnnSeqCrfTagger(
        rnn2seqencoder=encoder,
        encoding_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        datasets_manager=dataset_manager,
    )

    return (
        tagger,
        dataset_manager,
        {
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "HIDDEN_DIM": 2 * HIDDEN_DIM
            if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
            else HIDDEN_DIM,
            "COMBINE_STRATEGY": COMBINE_STRATEGY,
            "BIDIRECTIONAL": BIDIRECTIONAL,
            "EXPECTED_HIDDEN_DIM": 2 * HIDDEN_DIM
            if COMBINE_STRATEGY == "concat" and BIDIRECTIONAL
            else HIDDEN_DIM,
        },
    )


class TestParscitTagger:
    def test_parscit_tagger_namespaces(self, setup_parscit_tagger, seq_dataset_manager):
        tagger, dataset_manager, options = setup_parscit_tagger
        dataset_manager = seq_dataset_manager
        lines, labels = dataset_manager.train_dataset.get_lines_labels()

        output_dict = tagger(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )

        assert "logits_seq_label" in output_dict.keys()
        assert "predicted_tags_seq_label" in output_dict.keys()
        assert "loss" in output_dict.keys()

    def test_parscit_tagger_dimensions(self, setup_parscit_tagger, seq_dataset_manager):
        tagger, dataset_manager, options = setup_parscit_tagger
        dataset_manager = seq_dataset_manager
        lines, labels = dataset_manager.train_dataset.get_lines_labels()
        output_dict = tagger(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )
        assert output_dict["logits_seq_label"].size() == (2, 3, 7)
