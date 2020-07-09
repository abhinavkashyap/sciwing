import click
import pathlib
import sciwing.constants as constants
from sciwing.utils.common import download_file


DATA_FILE_URLS = constants.DATA_FILE_URLS


@click.group()
def download():
    """ Download group of commands that helps in downloading to the user machine
    """
    pass


@download.command()
@click.option(
    "--task",
    type=click.Choice(["sectlabel", "genericsect", "scienceie", "cit_int_clf"]),
    help="Chose from one of the following tasks",
)
@click.option(
    "--path", default=".", help="The directory where the data will be downloaded"
)
def data(task, path):
    """ Downloads the data for a particular task.
    """
    path = pathlib.Path(path)
    if task == "sectlabel":
        urls = [
            DATA_FILE_URLS["SECT_LABEL_TRAIN_FILE"],
            DATA_FILE_URLS["SECT_LABEL_DEV_FILE"],
            DATA_FILE_URLS["SECT_LABEL_TEST_FILE"],
        ]
        for url in urls:
            dest_filename = path.joinpath(url.split("/")[-1])
            if not dest_filename.is_file():
                download_file(url=url, dest_filename=dest_filename)

    if task == "genericsect":
        urls = [
            DATA_FILE_URLS["GENERIC_SECTION_TRAIN_FILE"],
            DATA_FILE_URLS["GENERIC_SECTION_DEV_FILE"],
            DATA_FILE_URLS["GENERIC_SECTION_TEST_FILE"],
        ]
        for url in urls:
            dest_filename = path.joinpath(url.split("/")[-1])
            if not dest_filename.is_file():
                download_file(url=url, dest_filename=dest_filename)

    if task == "scienceie":
        train_file_url = DATA_FILE_URLS["TRAIN_SCIENCE_IE_CONLL_FILE"]
        train_dest_filename = path.joinpath("train_science_ie_conll.txt")
        if not train_dest_filename.is_file():
            download_file(url=train_file_url, dest_filename=train_dest_filename)

        dev_file_url = DATA_FILE_URLS["DEV_SCIENCE_IE_CONLL_FILE"]
        dev_dest_filename = path.joinpath("dev_science_ie_conll.txt")
        if not dev_dest_filename.is_file():
            download_file(url=dev_file_url, dest_filename=dev_dest_filename)

    if task == "parscit":
        urls = [
            DATA_FILE_URLS["PARSCIT_TRAIN_FILE"],
            DATA_FILE_URLS["PARSCIT_DEV_FILE"],
            DATA_FILE_URLS["PARSCIT_TEST_FILE"],
        ]
        for url in urls:
            dest_filename = path.joinpath(url.split("/")[-1])
            if not dest_filename.is_file():
                download_file(url=url, dest_filename=dest_filename)

    if task == "cit_int_clf":
        urls = [
            DATA_FILE_URLS["SCICITE_TRAIN"],
            DATA_FILE_URLS["SCICITE_DEV"],
            DATA_FILE_URLS["SCICITE_TEST"],
        ]
        for url in urls:
            dest_filename = path.joinpath(url.split("/")[-1])
            if not dest_filename.is_file():
                download_file(url=url, dest_filename=dest_filename)
