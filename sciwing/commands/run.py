import click
from sciwing.utils.sciwing_toml_runner import SciWingTOMLRunner
import pathlib


@click.command()
@click.argument("toml_filename")
def run(toml_filename):
    """Given a toml filename where the dataset, model and engine are defined
    this command creates the model, runs it and reports the results on test dataset

    Parameters
    ----------
    toml_filename: filename
        Full path of the toml filename

    Returns
    -------
    None
        Runs the model and prints the results.

    """
    toml_filepath = pathlib.Path(toml_filename)
    if not toml_filepath.is_file():
        raise FileNotFoundError(f"TOML File {toml_filename} is not found")

    sciwing_toml_runner = SciWingTOMLRunner(toml_filename=pathlib.Path(toml_filename))
    sciwing_toml_runner.run()
