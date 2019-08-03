import pytest
from parsect.infer.parscit_inference import ParscitInference
from parsect.datasets.seq_labeling.parscit_dataset import ParscitDataset
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.embedders.concat_embedders import ConcatEmbedders
from parsect.modules.charlstm_encoder import CharLSTMEncoder
from parsect.models.parscit_tagger import ParscitTagger
import parsect.constants as constants
import pathlib
import json
import torch.nn as nn
import torch
from parsect.infer.bilstm_crf_infer import get_bilstm_crf_infer

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture(scope="session")
def setup_base_parscit_inference():
    debug_parscit_model_folder = pathlib.Path(OUTPUT_DIR, "lstm_crf_parscit_debug")
    inference_client = get_bilstm_crf_infer(str(debug_parscit_model_folder))
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
