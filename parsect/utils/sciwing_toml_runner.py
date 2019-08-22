import pathlib
import toml
from parsect.utils.exceptions import TOMLConfigurationError
from parsect.datasets import *
from parsect.models import *
from parsect.metrics import *
from parsect.engine import *
from parsect.modules import *
from parsect.utils.class_nursery import ClassNursery
from parsect.utils.common import create_class
import torch.nn as nn
import inspect


class SciWingTOMLRunner:
    def __init__(self, toml_filename: pathlib.Path):
        self.toml_filename = toml_filename
        self.doc = self._parse_toml_file()

        self.experiment_name = None
        self.all_datasets = (
            None
        )  # Dict {'train': Dataset, 'valid': Dataset, 'test': Dataset}
        self.model = None
        self.engine = None
        self._parse()

    def _parse_toml_file(self):
        try:
            with open(self.toml_filename) as fp:
                doc = toml.load(fp)
                return doc
        except FileNotFoundError:
            print(f"File {self.toml_filename} is not found")

    def _parse(self):

        # experiment section
        experiment_section = self.doc.get("experiment")
        self.experiment_name = experiment_section.get("exp_name")
        self.experiment_dir = pathlib.Path(experiment_section.get("exp_dir"))

        if self.experiment_dir.is_dir():
            raise FileExistsError(f"{self.experiment_dir} already exists")
        else:
            self.experiment_dir.mkdir()

        # get the dataset section from toml
        dataset_section = self.doc.get(f"dataset")
        if dataset_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have a datasets section. Please "
                f"Provide a dataset section in your toml file"
            )
        else:
            self.all_datasets = self.parse_dataset_section()

        # get the model section from toml
        model_section = self.doc.get("model")
        if model_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have model section."
                f"Please provide a model section to construct the model"
            )
        else:
            self.model = self.parse_model_section()

        # get the engine section from toml
        engine_section = self.doc.get("engine")
        if engine_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have an engine section"
            )
        else:
            self.engine = self.parse_engine_section()

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
                module_classname = section[key]["class"]
                cls_obj = create_class(
                    module_name=ClassNursery.class_nursery[module_classname],
                    classname=section[key]["class"],
                )
                submodule = cls_obj(**insider_args)
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
                    else:
                        classname = subsection.get("class")
                        embedder_args = {}
                        for attr_key, attr in subsection.items():
                            if attr_key in ["class", "name"]:
                                pass
                            else:
                                embedder_args[attr_key] = attr
                        cls_obj = create_class(
                            classname=classname,
                            module_name=ClassNursery.class_nursery[classname],
                        )
                        embedder = cls_obj(**embedder_args)
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

    def parse_engine_section(self):
        engine_section = self.doc.get("engine")
        engine_args = {}
        for key, value in engine_section.items():
            if not isinstance(value, dict):
                engine_args[key] = value

        optimizer_section = engine_section.get("optimizer")

        if optimizer_section is None:
            optimizer_classname = "SGD"
            optimizer_module = ClassNursery.class_nursery[optimizer_classname]
            optimizer_args = {"lr": 1e-2}
        else:
            optimizer_classname = optimizer_section.get("class")
            optimizer_module = ClassNursery.class_nursery[optimizer_classname]
            optimizer_args = {}
            for arg_, value in optimizer_section.items():
                if arg_ != "class":
                    optimizer_args[arg_] = value

        optimizer_cls = create_class(
            module_name=optimizer_module, classname=optimizer_classname
        )
        optimizer = optimizer_cls(params=self.model.parameters(), **optimizer_args)
        # patching optimizer
        engine_args["optimizer"] = optimizer

        metric_section = engine_section.get("metric")
        metric_classname = metric_section.get("class")
        metric_args = {}
        for key, value in metric_section.items():
            if key == "class":
                pass
            else:
                metric_args[key] = value

        metric_cls = create_class(
            module_name=ClassNursery.class_nursery[metric_classname],
            classname=metric_classname,
        )
        metric = metric_cls(**metric_args)
        engine_args["metric"] = metric

        train_dataset = self.all_datasets["train"]
        valid_dataset = self.all_datasets["valid"]
        test_dataset = self.all_datasets["test"]
        engine_args["train_dataset"] = train_dataset
        engine_args["validation_dataset"] = valid_dataset
        engine_args["test_dataset"] = test_dataset
        engine_args["model"] = self.model
        engine_args["experiment_name"] = self.experiment_name
        engine_args["experiment_hyperparams"] = self.doc

        engine_module = ClassNursery.class_nursery["Engine"]
        engine_classname = "Engine"
        engine_cls = create_class(classname=engine_classname, module_name=engine_module)
        engine = engine_cls(**engine_args)
        return engine

    def run(self):
        self.engine.run()


if __name__ == "__main__":
    import parsect.constants as constants

    PATHS = constants.PATHS
    CONFIGS_DIR = PATHS["CONFIGS_DIR"]
    bow_random_parsect_toml = pathlib.Path(CONFIGS_DIR, "bow_random_parsect.toml")
    toml_parser = SciWingTOMLRunner(toml_filename=bow_random_parsect_toml)
