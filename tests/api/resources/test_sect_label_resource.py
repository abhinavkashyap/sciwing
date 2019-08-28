import pytest
import requests
from sciwing.api.pdf_store import PdfStore
from sciwing.api.resources.sect_label_resource import SectLabelResource
import pathlib
import sciwing.api.conf as config

BIN_FOLDER = config.BIN_FOLDER
jar_path = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")


class TestSectLabelResource:
    def test_file_deleted_after_request(self, tmpdir):
        # Test file is deleted after request
        binary_string = b"Some text"
        pdf_store_dir = tmpdir.mkdir("pdf_store")
        pdf_store_dir = pathlib.Path(pdf_store_dir)
        pdf_store = PdfStore(store_path=pdf_store_dir)
        pdf_store.save_pdf_binary_string(
            pdf_string=binary_string, out_filename="temp_pdf.pdf"
        )

        # mock the post request

        # check if the file is deleted after the response returns
