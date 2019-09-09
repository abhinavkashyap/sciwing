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
@click.option("--task", type=click.Choice(["sectlabel", "genericsect"]))
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
