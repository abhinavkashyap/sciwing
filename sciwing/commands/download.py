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
@click.option("--task", type=click.Choice(["sectlabel", "genericsect", "scienceie"]))
@click.option("--path", default=".")
def data(task, path):
    path = pathlib.Path(path)
    if task == "sectlabel":
        url = DATA_FILE_URLS["SECT_LABEL_FILE"]
        dest_filename = path.joinpath("sectLabel.train.data")
        if not dest_filename.is_file():
            download_file(url=url, dest_filename=dest_filename)

    if task == "genericsect":
        url = DATA_FILE_URLS["GENERIC_SECTION_TRAIN_FILE"]
        dest_filename = path.joinpath("genericSect.train.data")
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
