import pathlib
import sciwing.api.conf as config
import subprocess

BIN_FOLDER = config.BIN_FOLDER
PDF_BOX_JAR = BIN_FOLDER.joinpath("pdfbox-app-2.0.16.jar")


class PdfReader:
    def __init__(self, filepath: pathlib.Path):
        """ Reads the pdf file, performs cleaning and transformations

        Parameters
        ----------
        filepath : pathlib.Path
            The path to read the pdf file
        """
        self.filepath = filepath
        self.pdf_box_jar = PDF_BOX_JAR

    def read_pdf(self):
        text = subprocess.run(
            ["java", "-jar", PDF_BOX_JAR, "ExtractText", "-console", self.filepath],
            stdout=subprocess.PIPE,
        )
        text = text.stdout
        text = text.decode("utf-8")
        text = text.split("\n")
        return text
