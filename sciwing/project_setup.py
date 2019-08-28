import wasabi
import pathlib
from google_drive_downloader import GoogleDriveDownloader as gdd
from sciwing.utils.common import download_file, extract_zip

import sciwing.constants as constants

FILES = constants.FILES
PATHS = constants.PATHS
SECT_LABEL_FILE_GID = FILES["SECT_LABEL_FILE_GID"]
GLOVE_FILE = FILES["GLOVE_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


def project_setup():
    msg_printer = wasabi.Printer()
    msg_printer.divider("Settting up sciwing for development")

    msg_printer.info("Creating data directory")
    data_dir_path = pathlib.Path(DATA_DIR)
    try:
        data_dir_path.mkdir()
        embeddings_dir = pathlib.Path(data_dir_path, "embeddings")
        glove_embeddings_dir = pathlib.Path(embeddings_dir, "glove")
        outputs_dir = pathlib.Path("./outputs")
        embeddings_dir.mkdir()
        glove_embeddings_dir.mkdir()
        outputs_dir.mkdir()

        msg_printer.good("Created data and embeddings folder")

        # download the sectLabel file to data dir
        with msg_printer.loading("Downloading sect label file"):
            gdd.download_file_from_google_drive(
                file_id=SECT_LABEL_FILE_GID,
                dest_path=str(data_dir_path) + "/sectLabel.train.data",
                unzip=False,
            )
        msg_printer.good("Downloaded sect label file")

        # download glove embeddings
        download_file(GLOVE_FILE, str(glove_embeddings_dir))
        extract_zip(str(glove_embeddings_dir) + "/glove.6B.zip", glove_embeddings_dir)

    except FileExistsError:
        msg_printer.fail(f"The path {data_dir_path} already exists")


if __name__ == "__main__":
    project_setup()
