import falcon
from typing import Dict, Callable
from parsect.api.pdf_store import PdfStore
import subprocess
import pathlib


class SectLabelResource:
    def __init__(self, pdf_store: PdfStore, pdfbox_jar_path: str):
        self.pdf_store = pdf_store
        self.pdfbox_jar_path = pdfbox_jar_path

    def on_post(self, req, resp) -> Dict[str, str]:
        """ Post the base64 url encoded pdf file to sect label

        Returns
        -------
        Dict[str, str]
            Return the lines with corresponding labels to the client
        """
        file = req.get_param("file", None)
        if file is None:
            resp.status = falcon.HTTP_400
            resp.body = f"File not found in your request."

        else:
            file_contents = file.file.read()  # binary string
            filename = file.filename
            pdf_save_location = self.pdf_store.save_pdf_binary_string(
                pdf_string=file_contents, out_filename=filename
            )

            # run pdfbox to get the lines from the file
            # running the jar file .. need to find a better solution
            text = subprocess.run(
                [
                    "java",
                    "-jar",
                    self.pdfbox_jar_path,
                    "ExtractText",
                    "-console",
                    pdf_save_location,
                ],
                capture_output=True,
            )
            text = text.stdout
            resp.data = text

        resp.status = falcon.HTTP_201
