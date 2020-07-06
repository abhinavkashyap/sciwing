from fastapi import APIRouter, UploadFile, File
from sciwing.pipelines.pipeline import pipeline
import sciwing.api.conf as config
from sciwing.api.utils.pdf_store import PdfStore
from sciwing.api.utils.pdf_reader import PdfReader


PDF_CACHE_DIR = config.PDF_STORE_LOCATION
BIN_FOLDER = config.BIN_FOLDER

if not PDF_CACHE_DIR.is_dir():
    PDF_CACHE_DIR.mkdir()


router = APIRouter()
pdf_store = PdfStore(PDF_CACHE_DIR)
PDF_BOX_JAR = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")
processing_pipeline = None


@router.post("/pdf_pipeline/uploadfile")
def pdf_pipeline(file: UploadFile = File(None)):
    file_handle = file.file
    file_name = file.filename
    file_contents = file_handle.read()
    global processing_pipeline

    pdf_save_location = pdf_store.save_pdf_binary_string(
        pdf_string=file_contents, out_filename=file_name
    )

    if not processing_pipeline:
        processing_pipeline = pipeline("pdf_pipeline")

    ents = processing_pipeline(pdf_save_location)
    response = {}
    for name, ent in ents["ents"].items():
        response[name] = ent
    return response
