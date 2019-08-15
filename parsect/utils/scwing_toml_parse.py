import pathlib
import toml
from parsect.utils.exceptions import TOMLConfigurationError
from parsect.datasets import *
import copy
from parsect.utils.class_nursery import ClassNursery
from parsect.utils.common import create_class


class SciWingTOMLParse:
    def __init__(self, toml_filename: pathlib.Path):
        self.toml_filename = toml_filename
        self.doc = self._parse_toml_file()
        self._parse()

    def _parse_toml_file(self):
        try:
            with open(self.toml_filename) as fp:
                doc = toml.load(fp)
                return doc
        except FileNotFoundError:
            print(f"File {self.toml_filename} is not found")

    def _parse(self):
        dataset = self.doc.get("dataset")
        if dataset is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have a dataset section. Please "
                f"Provide a dataset section in your toml file"
            )
        self.parse_dataset_section()

    def parse_dataset_section(self):
        dataset = self.doc.get("dataset")

        dataset_name = dataset.get("name")
        dataset_classname = dataset.get("class")
        if dataset_name is None or dataset_classname is None:
            raise TOMLConfigurationError(
                f"Dataset section needs to have a name and class section"
            )
        args = dataset.get("args")
        try:
            dataset_cls = create_class(
                classname=dataset_classname,
                module_name=ClassNursery.class_nursery[dataset_classname],
            )
            dataset = dataset_cls(**args)
        except ModuleNotFoundError:
            print(
                f"Module {ClassNursery.class_nursery[dataset_classname]} is not found"
            )
        except AttributeError:
            print(f"Class {dataset_classname} is not found ")


if __name__ == "__main__":
    import parsect.constants as constants

    PATHS = constants.PATHS
    CONFIGS_DIR = PATHS["CONFIGS_DIR"]
    bow_random_parsect_toml = pathlib.Path(CONFIGS_DIR, "bow_random_parsect.toml")
    toml_parser = SciWingTOMLParse(toml_filename=bow_random_parsect_toml)
