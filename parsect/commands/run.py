import click
from parsect.utils.scwing_toml_parse import SciWingTOMLRunner
import pathlib


@click.command()
@click.argument("toml_filename")
def run(toml_filename):
    toml_filepath = pathlib.Path(toml_filename)
    if not toml_filepath.is_file():
        raise FileNotFoundError(f"TOML File {toml_filename} is not found")

    sciwing_toml_runner = SciWingTOMLRunner(toml_filename=pathlib.Path(toml_filename))
    sciwing_toml_runner.run()
