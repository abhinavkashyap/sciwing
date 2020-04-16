import pathlib


class PdfStore:
    def __init__(self, store_path: pathlib.Path):
        """Manages the storage, retrieval and deletion of pdf files. This is useful
        when we have to manipulate pdf files and delete them eventually.

        Parameters
        ----------
        store_path : pathlib.Path
            The store path is where all the pdfs will be stored and deleted from
        """
        self.store_path = store_path

    def save_pdf_binary_string(
        self, pdf_string: bytes, out_filename: str
    ) -> pathlib.Path:
        """ Save the binary string to the store using the out_filename

        Parameters
        ----------
        pdf_string : str
            String representing a pdf in binary format

        out_filename: str
            The name of the pdf file that will stored

        Returns
        -------
        None

        """
        pdf_filename = self.store_path.joinpath(out_filename)
        with open(pdf_filename, "wb") as fp:
            fp.write(pdf_string)

        return pdf_filename

    def retrieve_binary_string_from_store(self, filename: str):
        """ Retrieve the contents of the pdf file as binary from the store

        Parameters
        ----------
        filename : str
            Filename from which the file should be read

        Returns
        -------
        bytes
            A binary string of the filename
        """
        filepath = self.store_path.joinpath(filename)

        if not filepath.is_file():
            raise ValueError(
                f"Filename {filename} you requested is not present in the store"
            )

        with open(filepath, "rb") as fp:
            string = fp.read()

        return string

    def delete_file(self, filename: str):
        """ Deletes the files from the store

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """

        filepath = self.store_path.joinpath(filename)

        if not filepath.is_file():
            raise ValueError(f"File name {filepath.name} does not exist in the store")

        filepath.unlink()
