import pathlib
import parsect.constants as constants
from parsect.infer.bow_lc_parsect_infer import get_bow_lc_parsect_infer
from parsect.infer.bilstm_crf_infer import get_bilstm_crf_infer

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]

PDF_STORE_LOCATION = pathlib.Path("/tmp/")
BIN_FOLDER = pathlib.Path("./bin/")

# TODO: This should come from the cloud service
SECT_LABEL_MODEL_PATH = pathlib.Path(OUTPUT_DIR, "parsect_bow_random_emb_lc_debug")
SECT_LABEL_INFER_FUNCTION = get_bow_lc_parsect_infer

PARSCIT_TAGGER_MODEL_PATH = pathlib.Path(OUTPUT_DIR, "lstm_crf_parscit_debug")
PARSCIT_TAGGER_INFER_FUNCTION = get_bilstm_crf_infer
