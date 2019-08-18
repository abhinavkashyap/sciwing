import pathlib
import toml
from parsect.utils.exceptions import TOMLConfigurationError
from parsect.datasets import *
from parsect.models import *
from parsect.modules import *
import copy
import torch.nn as nn
from parsect.utils.class_nursery import ClassNursery
from parsect.utils.common import create_class
import inspect
import torch.nn as nn


class SciWingTOMLParse:
    def __init__(self, toml_filename: pathlib.Path):
        self.toml_filename = toml_filename
        self.doc = self._parse_toml_file()
        self.all_datasets = (
            None
        )  # Dict {'train': Dataset, 'valid': Dataset, 'test': Dataset}
        self._parse()

    def _parse_toml_file(self):
        try:
            with open(self.toml_filename) as fp:
                doc = toml.load(fp)
                return doc
        except FileNotFoundError:
            print(f"File {self.toml_filename} is not found")

    def _parse(self):

        # get the dataset section from toml
        dataset_section = self.doc.get(f"dataset")
        if dataset_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have a datasets section. Please "
                f"Provide a dataset section in your toml file"
            )
        self.all_datasets = self.parse_dataset_section()

        # get the model section from toml
        model_section = self.doc.get("model")

        if model_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have model secction."
                f"Please provide a model section to construct the model"
            )
        else:
            self.parse_model_section()

    def parse_dataset_section(self):
        dataset_section = self.doc.get("dataset")
        all_datasets = {}

        dataset_name = dataset_section.get("name")
        dataset_classname = dataset_section.get("class")
        if dataset_name is None or dataset_classname is None:
            raise TOMLConfigurationError(
                f"Dataset section needs to have a name and class section"
            )
        args = dataset_section.get("args")
        for dataset_type in ["train", "valid", "test"]:
            try:
                dataset_cls = create_class(
                    classname=dataset_classname,
                    module_name=ClassNursery.class_nursery[dataset_classname],
                )
                args["dataset_type"] = dataset_type
                args["filename"] = dataset_section[f"{dataset_type}_filename"]
                dataset = dataset_cls(**args)
                all_datasets[dataset_type] = dataset
            except ModuleNotFoundError:
                print(
                    f"Module {ClassNursery.class_nursery[dataset_classname]} is not found"
                )
            except AttributeError:
                print(f"Class {dataset_classname} is not found ")
        return all_datasets

    def _parse_model_section(self, section: dict, args={}):
        for key in section.keys():
            value = section[key]
            if key in ["class", "name", "self"]:
                pass

            # we will instantiate the submodules in this
            elif isinstance(value, dict):
                insider_args = self._parse_model_section(section[key], args={})
                print("insider_args", insider_args)
                module_classname = section[key]["class"]
                cls_obj = create_class(
                    module_name=ClassNursery.class_nursery[module_classname],
                    classname=section[key]["class"],
                )
                submodule = cls_obj(**insider_args)
                print(f"submodule {submodule}")
                print(f"insider arg {insider_args}")
                args[key] = submodule
            # special handling for embedders
            elif isinstance(value, list):
                embedders = []
                for subsection in value:
                    if subsection.get("embed") == "word_vocab":
                        is_freeze = subsection.get("freeze", False)
                        embedding = self.all_datasets[
                            "train"
                        ].word_vocab.load_embedding()
                        embedding = nn.Embedding.from_pretrained(
                            embeddings=embedding, freeze=is_freeze
                        )
                        embedding_dim = self.all_datasets[
                            "train"
                        ].word_vocab.embedding_dimension
                        embedder = VanillaEmbedder(
                            embedding_dim=embedding_dim, embedding=embedding
                        )
                        embedders.append(embedder)
                final_embedder = ConcatEmbedders(embedders=embedders)
                args["embedder"] = final_embedder

            else:
                args[key] = section[key]
        return args

    def parse_model_section(self):
        """ Parses the Model section of the toml file

            Returns
            -------
            nn.Module
                A torch module representing the model
            """
        model_section = self.doc.get("model")
        # parse all the different arguments
        args = self._parse_model_section(section=model_section)
        model_classname = model_section.get("class")
        model_module_name = ClassNursery.class_nursery.get(model_classname)
        cls_obj = create_class(classname=model_classname, module_name=model_module_name)
        model = cls_obj(**args)
        return model


if __name__ == "__main__":
    import parsect.constants as constants

    PATHS = constants.PATHS
    CONFIGS_DIR = PATHS["CONFIGS_DIR"]
    bow_random_parsect_toml = pathlib.Path(CONFIGS_DIR, "bow_random_parsect.toml")
    toml_parser = SciWingTOMLParse(toml_filename=bow_random_parsect_toml)
