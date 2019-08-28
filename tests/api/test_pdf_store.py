import pytest
from sciwing.api.pdf_store import PdfStore
import pathlib


@pytest.fixture
def setup_pdf_store(tmpdir):
    pdf_store_dir = tmpdir.mkdir("pdf_store")
    pdf_store_dir = pathlib.Path(pdf_store_dir)
    assert pdf_store_dir.is_dir()
    pdf_store = PdfStore(pdf_store_dir)
    return pdf_store


class TestPdfStore:
    def test_file_is_saved(self, setup_pdf_store):
        pdf_store = setup_pdf_store
        binary_string = b"dummy text"
        out_filename = "dummy_file.pdf"
        pdf_store.save_pdf_binary_string(
            pdf_string=binary_string, out_filename=out_filename
        )
        assert pdf_store.store_path.joinpath(out_filename).is_file()

    def test_no_file_raises_error(self, setup_pdf_store):
        pdf_store = setup_pdf_store
        filename = "doesnotexist.pdf"

        with pytest.raises(ValueError):
            pdf_store.retrieve_binary_string_from_store(filename=filename)

    def test_file_can_be_read(self, setup_pdf_store):
        pdf_store = setup_pdf_store
        binary_string = b"dummy text"
        out_filename = "dummy_file.pdf"
        pdf_store.save_pdf_binary_string(
            pdf_string=binary_string, out_filename=out_filename
        )
        string = pdf_store.retrieve_binary_string_from_store(filename=out_filename)
        assert string == binary_string

    def test_delete_non_existing_file(self, setup_pdf_store):
        pdf_store = setup_pdf_store
        filename = "doesnotexist.pdf"

        with pytest.raises(ValueError):
            pdf_store.delete_file(filename)

    def test_file_can_be_deleted(self, setup_pdf_store):
        pdf_store = setup_pdf_store
        binary_string = b"dummy text"
        out_filename = "dummy_file.pdf"
        pdf_store.save_pdf_binary_string(
            pdf_string=binary_string, out_filename=out_filename
        )
        pdf_store.delete_file(out_filename)

        assert pdf_store.store_path.joinpath(out_filename).is_file() == False
