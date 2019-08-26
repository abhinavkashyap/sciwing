import click
from parsect.utils.sciwing_toml_runner import SciWingTOMLRunner
from parsect.infer.classification.classification_inference import (
    ClassificationInference,
)
import pathlib

class_infer_client_mapping = {"SimpleClassifier": ClassificationInference}


@click.command()
@click.argument("toml_filename")
def test(toml_filename):
    """ Runs the test data using the toml filename and prints test dataset results

    Parameters
    ----------
    toml_filename : str
        TOML filename that defines the dataset model and engine

    Returns
    -------
    None
        Prints resuts on the test dataset

    """
    toml_filepath = pathlib.Path(toml_filename)
    if not toml_filepath.is_file():
        raise FileNotFoundError(f"TOML file {toml_filename} is not found")

    sciwing_toml_runner = SciWingTOMLRunner(
        toml_filename=pathlib.Path(toml_filename), infer=True
    )
    sciwing_toml_runner.parse()

    exp_dirpath = sciwing_toml_runner.experiment_dir
    exp_dirpath = pathlib.Path(exp_dirpath)
    model_filepath = exp_dirpath.joinpath("checkpoints", "best_model.pt")
    model = sciwing_toml_runner.model
    dataset = sciwing_toml_runner.all_datasets["test"]
    model_class = sciwing_toml_runner.model_section.get("class")
    inference_cls = class_infer_client_mapping.get(model_class)
    inference_client = inference_cls(
        model=model, model_filepath=model_filepath, dataset=dataset
    )
    inference_client.run_test()
    inference_client.print_metrics()
