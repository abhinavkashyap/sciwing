import click
from sciwing.utils.sciwing_toml_runner import SciWingTOMLRunner
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
import pathlib

class_infer_client_mapping = {"SimpleClassifier": ClassificationInference}


@click.command()
@click.argument("toml_filename")
def test(toml_filename):
    """ Loads the model from experiment directory declared in toml filename. Runs
    test dataset against the model and reports the results.

    Parameters
    ----------
    toml_filename : str
        TOML filename that defines the dataset model and engine

    Returns
    -------
    None
        Reports resuts on the test dataset

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
    data_manager = sciwing_toml_runner.datasets_manager
    model_class = sciwing_toml_runner.model_section.get("class")
    inference_cls = class_infer_client_mapping.get(model_class)
    inference_client = inference_cls(
        model=model, model_filepath=model_filepath, datasets_manager=data_manager
    )
    inference_client.run_test()
    inference_client.report_metrics()
