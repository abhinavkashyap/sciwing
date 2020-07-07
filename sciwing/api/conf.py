import pathlib
import sciwing.constants as constants
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]

PDF_STORE_LOCATION = pathlib.Path("/tmp/sciwing_pdf_cache")
BIN_FOLDER = pathlib.Path(CURRENT_DIR, "bin")
