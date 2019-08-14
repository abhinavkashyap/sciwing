import pathlib
import toml
from parsect.utils.exceptions import TOMLConfigurationError
from parsect.datasets.sprinkle_dataset import sprinkle_dataset


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
        dataset_class = dataset.get("class")
        if dataset_name is None or dataset_class is None:
            raise TOMLConfigurationError(
                f"Dataset section needs to have a name and class section"
            )
        args = dataset.get("args")


if __name__ == "__main__":
    import parsect.constants as constants

    PATHS = constants.PATHS
    CONFIGS_DIR = PATHS["CONFIGS_DIR"]
    bow_random_parsect_toml = pathlib.Path(CONFIGS_DIR, "bow_random_parsect.toml")
    toml_parser = SciWingTOMLParse(toml_filename=bow_random_parsect_toml)
