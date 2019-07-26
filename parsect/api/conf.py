import pathlib
import parsect.constants as constants
from parsect.infer.bow_glove_emb_lc_parsect_infer import get_glove_emb_lc_parsect_infer

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]

PDF_STORE_LOCATION = pathlib.Path("/tmp/")
BIN_FOLDER = pathlib.Path("./bin/")
SECT_MODEL_PATH = pathlib.Path(
    OUTPUT_DIR, "parsect_bow_glove_emb_lc_3kw_10ml_100d_50e_1e-4lr"
)
SECT_LABEL_INFER_FUNCTION = get_glove_emb_lc_parsect_infer
