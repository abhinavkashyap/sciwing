from fastapi import APIRouter, UploadFile, File
from sciwing.models.sectlabel import SectLabel
from sciwing.api.utils.pdf_store import PdfStore
from sciwing.api.utils.pdf_reader import PdfReader
from sciwing.utils.common import chunks
import itertools
import sciwing.api.conf as config

PDF_CACHE_DIR = config.PDF_STORE_LOCATION
BIN_FOLDER = config.BIN_FOLDER

if not PDF_CACHE_DIR.is_dir():
    PDF_CACHE_DIR.mkdir(parents=True)

router = APIRouter()

sectlabel_model = None
pdf_store = PdfStore(PDF_CACHE_DIR)
PDF_BOX_JAR = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")


@router.post("/sectlabel/uploadfile/")
def process_pdf(file: UploadFile = File(None)):
    """ Classifies every line in the document to the logical section of the document. The logical
    section can be title, author, email, section header, subsection header etc

    Parameters
    ----------
    file : File
        The Bytestream of a file to be uploaded

    Returns
    -------
    JSON
        ``{"labels": [(line, label)]}``

    """
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
    pdf_reader = PdfReader(filepath=pdf_save_location)

    # read pdf lines
    lines = pdf_reader.read_pdf()
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

    # remove the saved pdf
    pdf_store.delete_file(str(pdf_save_location))

    return {"labels": response_tuples}


@router.post("/sectlabel/abstract/")
def extract_pdf(file: UploadFile = File(None)):
    """ Extracts the abstract from a scholarly article

    Parameters
    ----------
    file : uploadFile
        Byte Stream of a file uploaded.

    Returns
    -------
    JSON
        ``{"abstract": The abstract found in the scholarly document}``

    """

    global sectlabel_model
    if sectlabel_model is None:
        sectlabel_model = SectLabel()

    file_handle = file.file
    file_name = file.filename
    file_contents = file_handle.read()

    pdf_save_location = pdf_store.save_pdf_binary_string(
        pdf_string=file_contents, out_filename=file_name
    )

    print(f"pdf save location {pdf_save_location}")

    # noinspection PyTypeChecker
    pdf_reader = PdfReader(filepath=pdf_save_location)

    # read pdf lines
    lines = pdf_reader.read_pdf()
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

    abstract_lines = []
    found_abstract = False
    for line, label in response_tuples:
        if label == "sectionHeader" and line.strip().lower() == "abstract":
            found_abstract = True
            continue
        if found_abstract and label == "sectionHeader":
            break
        if found_abstract:
            abstract_lines.append(line.strip())

    abstract = " ".join(abstract_lines)

    # remove the saved pdf
    pdf_store.delete_file(str(pdf_save_location))

    return {"abstract": abstract}
