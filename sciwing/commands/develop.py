"""sciwing develop is a subcommand to setup the development environment
"""
import click
import pathlib
import sciwing.constants as constants
import wasabi
import zipfile
from sciwing.utils.common import download_file
from sciwing.utils.common import extract_zip
from sciwing.utils.common import extract_tar
import tarfile

DATA_FILE_URLS = constants.DATA_FILE_URLS
EMBEDDING_FILE_URLS = constants.EMBEDDING_FILE_URLS


@click.command()
@click.argument("action")
def develop(action):
    """ Making sciwing developer workflow easy

    You can pass either one of this as argument ["makedirs", "download"]
    makedirs creates appropriate directories in the developer environment
    download - downloads the appropriate folders.
    """
    HOME_DIR = pathlib.Path("~").expanduser()
    DATA_CACHE_DIR = HOME_DIR.joinpath(".sciwing.data_cache")
    MODEL_CACHE_DIR = HOME_DIR.joinpath(".sciwing.model_cache")
    EMBEDDING_CACHE_DIR = HOME_DIR.joinpath(".sciwing.embedding_cache")
    OUTPUT_CACHE_DIR = HOME_DIR.joinpath(".sciwing.output_cache")
    REPORTS_CACHE_DIR = HOME_DIR.joinpath(".sciwing.reports_cache")
    printer = wasabi.Printer()

    if action == "makedirs":
        if not DATA_CACHE_DIR.is_dir():
            DATA_CACHE_DIR.mkdir()
        printer.good(f"created {DATA_CACHE_DIR}")

        if not MODEL_CACHE_DIR.is_dir():
            MODEL_CACHE_DIR.mkdir()
        printer.good(f"created {MODEL_CACHE_DIR}")

        if not EMBEDDING_CACHE_DIR.is_dir():
            EMBEDDING_CACHE_DIR.mkdir()
        printer.good(f"created {EMBEDDING_CACHE_DIR}")

        if not OUTPUT_CACHE_DIR.is_dir():
            OUTPUT_CACHE_DIR.mkdir()
        printer.good(f"created {OUTPUT_CACHE_DIR}")

        if not REPORTS_CACHE_DIR.is_dir():
            REPORTS_CACHE_DIR.mkdir()
        printer.good(f"created {REPORTS_CACHE_DIR}")

    elif action == "download":
        for name, url in DATA_FILE_URLS.items():
            local_filename = pathlib.Path(url)
            local_filename = local_filename.parts[-1]
            local_filename = DATA_CACHE_DIR.joinpath(local_filename)

            if not local_filename.is_file():
                download_file(url, dest_filename=str(local_filename))

                if zipfile.is_zipfile(str(local_filename)):
                    extract_zip(
                        filename=str(local_filename),
                        destination_dir=str(local_filename.parent),
                    )
            else:
                printer.info(f"{local_filename} exists")

        for name, url in EMBEDDING_FILE_URLS.items():
            local_filename = pathlib.Path(url)
            local_filename = local_filename.parts[-1]
            local_filename = EMBEDDING_CACHE_DIR.joinpath(local_filename)

            if not local_filename.is_file():
                download_file(url, dest_filename=str(local_filename))

                if zipfile.is_zipfile(str(local_filename)):
                    extract_zip(
                        filename=str(local_filename),
                        destination_dir=str(local_filename.parent),
                    )
                if tarfile.is_tarfile(str(local_filename)):
                    if "tar" in local_filename.suffix:
                        mode = "r"
                    elif "gz" in local_filename.suffix:
                        mode = "r:gz"
                    else:
                        mode = "r"

                    extract_tar(
                        filename=str(local_filename),
                        destination_dir=str(local_filename.parent),
                        mode=mode,
                    )

            else:
                printer.info(f"{local_filename} exists")
