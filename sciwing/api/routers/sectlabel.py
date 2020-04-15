from fastapi import APIRouter, UploadFile, File
from sciwing.models.sectlabel import SectLabel
from sciwing.api.pdf_store import PdfStore
from sciwing.utils.common import chunks
import itertools
import subprocess
import sciwing.api.conf as config

PDF_CACHE_DIR = config.PDF_STORE_LOCATION
BIN_FOLDER = config.BIN_FOLDER

if not PDF_CACHE_DIR.is_dir():
    PDF_CACHE_DIR.mkdir()

router = APIRouter()

sectlabel_model = None
pdf_store = PdfStore(PDF_CACHE_DIR)
PDF_BOX_JAR = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")


@router.post("/sectlabel/uploadfile/")
def process_pdf(file: UploadFile = File(None)):
    global sectlabel_model
    if sectlabel_model is None:
        sectlabel_model = SectLabel()

    file_handle = file.file
    file_name = file.filename
    file_contents = file_handle.read()

    pdf_save_location = pdf_store.save_pdf_binary_string(
        pdf_string=file_contents, out_filename=file_name
    )
    # noinspection PyTypeChecker
    text = subprocess.run(
        ["java", "-jar", PDF_BOX_JAR, "ExtractText", "-console", pdf_save_location],
        stdout=subprocess.PIPE,
    )
    text = text.stdout
    text = str(text)
    lines = text.split("\\n")
    all_labels = []
    all_lines = []

    for batch_lines in chunks(lines, 64):
        labels = sectlabel_model.predict_for_text_batch(texts=batch_lines)
        all_labels.append(labels)
        all_lines.append(batch_lines)

    all_lines = itertools.chain.from_iterable(all_lines)
    all_lines = list(all_lines)

    all_labels = itertools.chain.from_iterable(all_labels)
    all_labels = list(all_labels)

    response_tuples = []
    for line, label in zip(all_lines, all_labels):
        response_tuples.append((line, label))

    return {"labels": response_tuples}
