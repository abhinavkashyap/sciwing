import pytest
import sciwing.constants as constants
import pathlib
from sciwing.infer.bow_lc_parsect_infer import get_bow_lc_parsect_infer
from sciwing.infer.bow_lc_gensect_infer import get_bow_lc_gensect_infer
from sciwing.infer.bow_elmo_emb_lc_parsect_infer import get_elmo_emb_lc_infer_parsect
from sciwing.infer.bow_elmo_emb_lc_gensect_infer import get_elmo_emb_lc_infer_gensect
from sciwing.infer.bow_bert_emb_lc_parsect_infer import (
    get_bow_bert_emb_lc_parsect_infer,
)
from sciwing.infer.bow_bert_emb_lc_gensect_infer import (
    get_bow_bert_emb_lc_gensect_infer,
)
from sciwing.infer.bi_lstm_lc_infer_parsect import get_bilstm_lc_infer_parsect
from sciwing.infer.bi_lstm_lc_infer_gensect import get_bilstm_lc_infer_gensect
from sciwing.infer.elmo_bi_lstm_lc_infer import get_elmo_bilstm_lc_infer

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
PATHS = constants.PATHS

OUTPUT_DIR = PATHS["OUTPUT_DIR"]

directory_infer_client_mapping = {
    "parsect_bow_random_emb_lc_debug": get_bow_lc_parsect_infer,
    "gensect_bow_random_emb_lc_debug": get_bow_lc_gensect_infer,
    "parsect_bow_elmo_emb_lc_debug": get_elmo_emb_lc_infer_parsect,
    "debug_gensect_bow_elmo_emb_lc": get_elmo_emb_lc_infer_gensect,
    "parsect_bow_bert_debug": get_bow_bert_emb_lc_parsect_infer,
    "gensect_bow_bert_debug": get_bow_bert_emb_lc_gensect_infer,
    "parsect_bi_lstm_lc_debug": get_bilstm_lc_infer_parsect,
    "debug_gensect_bi_lstm_lc": get_bilstm_lc_infer_gensect,
    "debug_parsect_elmo_bi_lstm_lc": get_elmo_bilstm_lc_infer,
}

directory_infer_client_mapping = directory_infer_client_mapping.items()


@pytest.fixture(params=directory_infer_client_mapping)
def setup_inference(tmpdir_factory, request):

    dirname = request.param[0]
    infer_func = request.param[1]
    debug_experiment_path = pathlib.Path(OUTPUT_DIR, dirname)

    inference_client = infer_func(str(debug_experiment_path))
    return inference_client


@pytest.mark.skipif(
    not all(
        [
            pathlib.Path(OUTPUT_DIR, dirname).exists()
            for dirname, _ in directory_infer_client_mapping
        ]
    ),
    reason="debug models do not exist in the output dir",
)
class TestClassificationInference:
    def test_run_inference_works(self, setup_inference):
        inference_client = setup_inference
        try:
            inference_client.run_inference()
        except:
            pytest.fail("Run inference for classification dataset fails")

    def test_run_test_works(self, setup_inference):
        inference_client = setup_inference
        try:
            inference_client.run_test()
        except:
            pytest.fail("Run test doest not work")

    def test_on_user_input_works(self, setup_inference):
        inference_client = setup_inference
        try:
            inference_client.on_user_input(line="test input")
        except:
            pytest.fail("On user input fails")

    def test_print_metrics_works(self, setup_inference):
        inference_client = setup_inference
        inference_client.run_test()
        try:
            inference_client.report_metrics()
        except:
            pytest.fail("Print metrics failed")

    def test_print_confusion_metrics_works(self, setup_inference):
        inference_client = setup_inference
        inference_client.run_test()
        try:
            inference_client.print_confusion_matrix()
        except:
            pytest.fail("Print confusion matrix fails")

    def test_get_misclassified_sentences(self, setup_inference):
        inference_client = setup_inference
        inference_client.run_test()
        try:
            inference_client.get_misclassified_sentences(
                true_label_idx=0, pred_label_idx=1
            )
        except:
            pytest.fail("Getting misclassified sentence fail")
