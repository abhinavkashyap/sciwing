import pytest
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.lstm2seqdecoder import Lstm2SeqDecoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.datasets.summarization.abstractive_text_summarization_dataset import AbstractiveSummarizationDatasetManager
from sciwing.models.simple_seq2seq import Seq2SeqModel


@pytest.fixture(scope="session")
def abs_sum_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train.txt")
    train_file.write(
        "word11_train word21_train###word11_label word21_label\nword12_train word22_train word32_train###word11_label word22_label"
    )

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev.txt")
    dev_file.write(
        "word11_dev word21_dev###word11_label word21_label\nword12_dev word22_dev word32_dev###word11_label word22_label"
    )

    test_file = tmpdir_factory.mktemp("test_data").join("test.txt")
    test_file.write(
        "word11_test word21_test###word11_label word21_label\nword12_test word22_test word32_test###word11_label word22_label"
    )

    data_manager = AbstractiveSummarizationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    return data_manager


@pytest.fixture()
def setup_seq2seq_model(abs_sum_dataset_manager):
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BIDIRECTIONAL = True
    COMBINE_STRATEGY = "concat"
    NUM_LAYERS = 1
    MAX_LENGTH = 4
    datasets_manager = abs_sum_dataset_manager

    embedder = WordEmbedder(embedding_type="glove_6B_50")

    encoder = Lstm2SeqEncoder(
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
        add_projection_layer=False,
    )

    decoder = Lstm2SeqDecoder(
        embedder=embedder,
        vocab=datasets_manager.namespace_to_vocab["tokens"],
        max_length=MAX_LENGTH,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        num_layers=NUM_LAYERS,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=False,
    )

    model = Seq2SeqModel(
        rnn2seqencoder=encoder,
        rnn2seqdecoder=decoder,
        datasets_manager=datasets_manager,
        enc_hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
    )

    return (
        model,
        datasets_manager,
        {
            "EMBEDDING_DIM": EMBEDDING_DIM,
            "MAX_LENGTH": MAX_LENGTH,
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


class TestSeq2SeqModel:
    def test_seq2seq_model_namespaces(self, setup_seq2seq_model, abs_sum_dataset_manager):
        model, dataset_manager, options = setup_seq2seq_model
        dataset_manager = abs_sum_dataset_manager
        lines, labels = dataset_manager.train_dataset.get_lines_labels()

        output_dict = model(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )

        assert len(lines) == 2
        assert "predicted_tags_tokens" in output_dict.keys()
        assert "loss" in output_dict.keys()

    def test_seq2seq_model_dimensions(self, setup_seq2seq_model, abs_sum_dataset_manager):
        model, dataset_manager, options = setup_seq2seq_model
        dataset_manager = abs_sum_dataset_manager
        VOCAB_SIZE = dataset_manager.namespace_to_vocab["tokens"].get_vocab_len()
        lines, labels = dataset_manager.train_dataset.get_lines_labels()
        output_dict = model(
            lines=lines,
            labels=labels,
            is_training=True,
            is_validation=False,
            is_test=False,
        )
        assert output_dict["predicted_probs_tokens"].size() == (2, options["MAX_LENGTH"], VOCAB_SIZE)
