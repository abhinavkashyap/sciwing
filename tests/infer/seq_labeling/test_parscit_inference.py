import pytest
import sciwing.constants as constants
import pathlib
from sciwing.infer.bilstm_crf_infer import get_bilstm_crf_infer

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


@pytest.fixture(scope="session")
def setup_base_parscit_inference():
    debug_parscit_model_folder = pathlib.Path(OUTPUT_DIR, "lstm_crf_parscit_debug")
    inference_client = get_bilstm_crf_infer(str(debug_parscit_model_folder))
    return inference_client


@pytest.mark.skipif(
    not pathlib.Path(OUTPUT_DIR, "lstm_crf_parscit_debug").exists(),
    reason="debug moel for lstm crf parscit does not exist",
)
class TestParscitInference:
    def test_run_inference_works(self, setup_base_parscit_inference):
        inference_client = setup_base_parscit_inference
        inference_client.run_test()
        assert type(inference_client.output_analytics) == dict

    def test_print_prf_table_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        inference.run_test()
        try:
            inference.print_metrics()
        except:
            pytest.fail("Parscit print prf table does not work")

    def test_print_confusion_metrics_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        inference.run_test()
        try:
            inference.print_confusion_matrix()
        except:
            pytest.fail("Parscit print confusion metrics fails")

    def test_generate_report_for_paper_works(self, setup_base_parscit_inference):
        inference = setup_base_parscit_inference
        inference.run_test()
        try:
            inference.generate_report_for_paper()
        except:
            pytest.fail("Parscit generate report for paper fails")

    def test_infer_single_sentence_works(self, setup_base_parscit_inference):
        inference_clinet = setup_base_parscit_inference
        try:
            inference_clinet.infer_single_sentence("A.B. Abalone, Future Paper")
        except:
            pytest.fail("Infer on single sentence does not work")
