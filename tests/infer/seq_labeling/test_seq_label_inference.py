import pytest
import sciwing.constants as constants
import pathlib
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from sciwing.engine.engine import Engine
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
import torch

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


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


@pytest.fixture(scope="session")
def setup_parscit_inference(seq_dataset_manager, tmpdir_factory):
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
        add_projection_layer=False,
    )

    tagger = RnnSeqCrfTagger(
        rnn2seqencoder=encoder,
        encoding_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        datasets_manager=dataset_manager,
    )

    train_metric = TokenClassificationAccuracy(datasets_manager=dataset_manager)
    dev_metric = TokenClassificationAccuracy(datasets_manager=dataset_manager)
    test_metric = TokenClassificationAccuracy(datasets_manager=dataset_manager)

    optimizer = torch.optim.Adam(params=tagger.parameters())
    batch_size = 1
    save_dir = tmpdir_factory.mktemp("experiment_1")
    num_epochs = 1
    save_every = 1
    log_train_metrics_every = 10

    engine = Engine(
        model=tagger,
        datasets_manager=dataset_manager,
        optimizer=optimizer,
        batch_size=batch_size,
        save_dir=save_dir,
        num_epochs=num_epochs,
        save_every=save_every,
        log_train_metrics_every=log_train_metrics_every,
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        track_for_best="macro_fscore",
    )

    engine.run()
    model_filepath = pathlib.Path(save_dir).joinpath("best_model.pt")

    inference_client = SequenceLabellingInference(
        model=tagger, model_filepath=model_filepath, datasets_manager=dataset_manager
    )

    return inference_client


class TestParscitInference:
    def test_run_inference_works(self, setup_parscit_inference):
        inference_client = setup_parscit_inference
        inference_client.run_test()
        assert isinstance(inference_client.output_analytics, dict)

    def test_print_prf_table_works(self, setup_parscit_inference):
        inference = setup_parscit_inference
        inference.run_test()
        try:
            inference.report_metrics()
        except:
            pytest.fail("Parscit print prf table does not work")

    def test_print_confusion_metrics_works(self, setup_parscit_inference):
        inference = setup_parscit_inference
        inference.run_test()
        try:
            inference.print_confusion_matrix()
        except:
            pytest.fail("Parscit print confusion metrics fails")

    def test_on_user_input(self, setup_parscit_inference):
        inference = setup_parscit_inference
        try:
            inference.on_user_input("A.B. Abalone, Future Paper")
        except:
            pytest.fail("Infer on single sentence does not work")

    def test_get_miscalssified_sentences(self, setup_parscit_inference):
        inference = setup_parscit_inference
        inference.run_test()
        try:
            inference.get_misclassified_sentences(true_label_idx=0, pred_label_idx=1)
        except:
            pytest.fail("Get misclassified sentences fails")
