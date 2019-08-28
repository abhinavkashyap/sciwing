import pytest
from sciwing.infer.science_ie_infer import get_science_ie_infer
import sciwing.constants as constants
import pathlib

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
FILES = constants.FILES
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]


@pytest.fixture(scope="session")
def setup_science_ie_inference():
    debug_parscit_model_folder = pathlib.Path(OUTPUT_DIR, "lstm_crf_scienceie_debug")
    return get_science_ie_infer(debug_parscit_model_folder)


@pytest.mark.skipif(
    not pathlib.Path(OUTPUT_DIR, "lstm_crf_science_ie_debug").exists(),
    reason="debug moel for lstm crf parscit does not exist",
)
class TestScienceIEInference:
    def test_print_prf_table_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            inference.print_prf_table()
        except:
            pytest.fail("Print PRF Table fails")

    def test_print_confusion_matrix_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            inference.print_confusion_matrix()
        except:
            pytest.fail("Print Confusion Matrix fails")

    @pytest.mark.parametrize("first_class", range(0, 8))
    @pytest.mark.parametrize("second_class", range(0, 8))
    def test_get_misclassified_sentences_works(
        self, setup_science_ie_inference, first_class, second_class
    ):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            inference.get_misclassified_sentences(first_class, second_class)
        except:
            pytest.fail("Getting Misclassified Sentences works")

    def test_generate_report_for_paper_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            inference.generate_report_for_paper()
        except:
            pytest.fail("Generate Report For paper does not work")

    def test_science_ie_user_input(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            inference.infer_single_sentence("This is a sentence")
        except:
            pytest.fail("On user input does not work")

    def test_scienceie_generate_pred_folder(
        self, setup_science_ie_inference, tmpdir_factory
    ):
        inference = setup_science_ie_inference
        inference.run_test()
        try:
            dev_folder = pathlib.Path(SCIENCE_IE_DEV_FOLDER)
            pred_folder = tmpdir_factory.mktemp("science_ie_pred_folder")
            pred_folder = pathlib.Path(pred_folder)
            inference.generate_predict_folder(
                dev_folder=dev_folder, pred_folder=pred_folder
            )
        except:
            pytest.fail("Generating Prediction Folder does not work")
