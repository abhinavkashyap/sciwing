import pytest
from parsect.infer.parscit_inference import ParscitInference
from parsect.datasets.parscit_dataset import ParscitDataset
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.models.parscit_tagger import ParscitTagger
import parsect.constants as constants
import pathlib
import json
import torch.nn as nn
import torch

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture
def setup_base_parscit_inference():
    debug_parscit_model_folder = pathlib.Path(OUTPUT_DIR, "lstm_crf_parscit_debug")
    hyperparam_config_filename = debug_parscit_model_folder.joinpath("config.json")
    test_conll_filepath = pathlib.Path(DATA_DIR, "parscit_test_conll.txt")

    with open(hyperparam_config_filename, "r") as fp:
        config = json.load(fp)

    MAX_NUM_WORDS = config.get("MAX_NUM_WORDS", None)
    MAX_LENGTH = config.get("MAX_LENGTH", None)
    VOCAB_STORE_LOCATION = config.get("VOCAB_STORE_LOCATION", None)
    DEBUG = config.get("DEBUG", None)
    DEBUG_DATASET_PROPORTION = config.get("DEBUG_DATASET_PROPORTION", None)
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", None)
    EMBEDDING_DIMENSION = config.get("EMBEDDING_DIMENSION", None)
    HIDDEN_DIMENSION = config.get("HIDDEN_DIM", None)
    BIDIRECTIONAL = config.get("BIDIRECTIONAL", None)
    COMBINE_STRATEGY = config.get("COMBINE_STRATEGY", None)
    DEVICE = config.get("DEVICE", "cpu")
    NUM_CLASSES = config.get("NUM_CLASSES", None)
    MODEL_SAVE_DIR = config.get("MODEL_SAVE_DIR", None)
    model_filepath = pathlib.Path(MODEL_SAVE_DIR, "best_model.pt")

    test_dataset = ParscitDataset(
        parscit_conll_file=test_conll_filepath,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
        start_token="<SOS>",
        end_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        add_start_end_token=False,
    )

    embedding = test_dataset.get_preloaded_embedding()
    embedding = nn.Embedding.from_pretrained(embedding)

    lstm2seqencoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIMENSION,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=True,
        device=torch.device(DEVICE),
    )
    model = ParscitTagger(
        rnn2seqencoder=lstm2seqencoder,
        num_classes=NUM_CLASSES,
        hid_dim=2 * HIDDEN_DIMENSION
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIMENSION,
    )

    inference_client = ParscitInference(
        model=model,
        model_filepath=str(model_filepath),
        hyperparam_config_filepath=str(hyperparam_config_filename),
        dataset=test_dataset,
    )
    return inference_client


class TestParscitInference:
    def test_run_inference_works(self, setup_base_parscit_inference):
        inference_client = setup_base_parscit_inference
        assert type(inference_client.output_analytics) == dict

    def test_print_prf_table_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        try:
            inference.print_prf_table()
        except:
            pytest.fail("Parscit print prf table does not work")

    def test_print_confusion_metrics_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        try:
            inference.print_confusion_matrix()
        except:
            pytest.fail("Parscit print confusion metrics fails")

    def test_generate_report_for_paper_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        try:
            inference.generate_report_for_paper()
        except:
            pytest.fail("Parscit generate report for paper fails")
