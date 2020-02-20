import pathlib
import toml
from sciwing.utils.exceptions import TOMLConfigurationError
from sciwing.datasets import *
from sciwing.models import *
from sciwing.metrics import *
from sciwing.engine import *
from sciwing.modules import *
from sciwing.utils.class_nursery import ClassNursery
from sciwing.utils.common import create_class
import torch.nn as nn
from typing import Dict
import networkx as nx
import copy
import wasabi
import pathlib
import sciwing.constants as constants

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


class SciWingTOMLRunner:
    def __init__(self, toml_filename: pathlib.Path, infer: bool = False):
        self.toml_filename = toml_filename
        self.infer = infer
        self.msg_printer = wasabi.Printer()
        self.doc = self._parse_toml_file()
        self.data_dir = pathlib.Path(DATA_DIR)

        self.experiment_name = None
        self.experiment_dir = None
        self.datasets_manager = None
        self.model_section = None
        self.dataset_section = None
        self.engine_section = None
        self.model = None
        self.engine = None
        self.model_dag = nx.DiGraph()

    def _parse_toml_file(self):
        """ Parses the toml file and returns the document

        Returns
        -------
        Dict[str, Any]
            The dictionary by parsing the toml file

        """
        try:
            with open(self.toml_filename) as fp:
                doc = toml.load(fp)
                return doc
        except FileNotFoundError:
            print(f"File {self.toml_filename} is not found")

    def parse(self):
        """ Parases the dataset, model and engine section of a toml file
        """

        # experiment section
        experiment_section = self.doc.get("experiment")
        self.experiment_name = experiment_section.get("exp_name")
        self.experiment_dir = pathlib.Path(experiment_section.get("exp_dir"))

        if not self.infer:
            if self.experiment_dir.is_dir():
                raise FileExistsError(f"{self.experiment_dir} already exists")
            else:
                self.experiment_dir.mkdir(parents=True)
        else:
            if not self.experiment_dir.is_dir():
                raise FileNotFoundError(
                    f"{self.experiment_dir} is not found for inference"
                )

        # get the dataset section from toml
        self.datasets_manager = self.parse_dataset_section()

        # get the model section from toml
        self.model = self.parse_model_section()

        # get the engine section from toml
        self.engine = self.parse_engine_section()

    def parse_dataset_section(self):
        """ Parse the dataset section of the toml file and instantiate the dataset

        Returns
        -------
        DatasetManager
            The dataset manager for the experiment
        """
        dataset_section = self.doc.get("dataset")
        if dataset_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have a datasets section. Please "
                f"Provide a dataset section in your toml file"
            )

        self.dataset_section = dataset_section
        dataset_classname = dataset_section.get("class")
        if dataset_classname is None:
            raise TOMLConfigurationError(
                f"Dataset section needs to have a name and class section"
            )

        train_filename = dataset_section.get("train_filename")
        dev_filename = dataset_section.get("dev_filename")
        test_filename = dataset_section.get("test_filename")
        train_filename = self.data_dir.joinpath(train_filename)
        dev_filename = self.data_dir.joinpath(dev_filename)
        test_filename = self.data_dir.joinpath(test_filename)
        train_filename = str(train_filename)
        dev_filename = str(dev_filename)
        test_filename = str(test_filename)

        tokenizers = dataset_section.get("tokenizers")
        namespace_vocab_options = dataset_section.get("namespace_vocab_options")
        namespace_numericalizer_map = dataset_section.get("namespace_numericalizer_map")

        args = {
            "train_filename": train_filename,
            "dev_filename": dev_filename,
            "test_filename": test_filename,
            "tokenizers": tokenizers,
            "namespace_vocab_options": namespace_vocab_options,
            "namespace_numericalizer_map": namespace_numericalizer_map,
        }
        try:
            dataset_cls = create_class(
                classname=dataset_classname,
                module_name=ClassNursery.class_nursery[dataset_classname],
            )
            dataset_manager = dataset_cls(**args)
            self.msg_printer.good(title=f"Finished Creating {dataset_classname}")
            return dataset_manager
        except ModuleNotFoundError:
            print(
                f"Module {ClassNursery.class_nursery[dataset_classname]} is not found"
            )
        except AttributeError:
            print(f"Class {dataset_classname} is not found ")

    def _form_dag(self, section_name: str, section: Dict, parent: str):
        """ Forms a DAG of the model section for execution

        The model can be a complex structure with various other sub-components that can be used
        One depends on the other and the order of execution has to be decided
        DAG is a good abstract model to define the dependence  between different modules
        This method instantiates a DAG given the section name, the TOML section that is being
        parsed with a directed edge between the parent and the child

        Parameters
        ----------
        section_name : str
            The name of the TOML section being parsed
        section : Dict
            The details of the actual section
        parent : str
            The node id of the parent graph

        Returns
        -------

        """
        classname = section.get("class")
        name = f"{section_name}__{classname}"
        self.model_dag.add_node(name)
        self.model_dag.nodes[name]["tag"] = section_name

        if parent is not None:
            self.model_dag.add_edge(parent, name)

        subsections = []
        for key, value in section.items():
            # another submodule is being used for this module in TOML
            if isinstance(value, dict):
                subsections.append({"section_name": key, "section": value})
            # This possibly is an embedder
            elif isinstance(value, list):
                for subsection in value:
                    subsections.append({"section_name": key, "section": subsection})
            else:
                self.model_dag.nodes[name][key] = value

        for subsection in subsections:
            self._form_dag(
                section_name=subsection["section_name"],
                section=subsection["section"],
                parent=name,
            )

    def _instantiate_model_using_dag(self):
        """ This is a key method that instantiates the DAG using topological order

        THE DAG from the TOML model section should be instantiated with the submodules
        of a module instantiated before the parent module can be instantiated
        This method does it using topological sort. Topoloogical sort is the sorting of
        nodes of a DAG where if there is an edge between two nodes from u ->v , then
        u appears before v in the ordering.

        We do exactly this for SciWING. We instantiate the children nodes that are
        used by parent nodes before we can instantiate the root node of the DAG
        that will represent the entire module.

        Returns
        -------
        nn.Module
            The instantiation of the root node

        """
        topo_order = nx.algorithms.topological_sort(self.model_dag)
        topo_order = reversed(list(topo_order))
        topo_order = list(topo_order)
        root_nodename = topo_order[-1]

        for node_id in topo_order:
            node_data = self.model_dag.nodes[node_id]
            tag = node_data.get("tag", None)
            classname = node_data.pop("class", None)
            if classname is None:
                raise TOMLConfigurationError(
                    f"Class {classname} is missing for one of the components of your model."
                    f"Have you added the class into the ClassNursery?"
                )
            class_args = copy.deepcopy(self.model_dag.nodes[node_id])

            # models accept a datasets manager by default
            # TODO: make this true for all the modules and models including embedders?
            if tag == "model" or tag == "embedder":
                class_args["datasets_manager"] = self.datasets_manager

            # instantiated class holds the object that is instantiated for
            # the node
            class_args.pop("instantiated_class", None)
            current_node_tag = class_args.pop("tag", None)

            # leaf node
            # we always assume that embedders are  used at the lower level
            # This is a reasonable assumption to make
            if not list(self.model_dag.successors(node_id)):
                cls_obj = create_class(
                    classname=classname,
                    module_name=ClassNursery.class_nursery[classname],
                )
                cls_obj = cls_obj(**class_args)
                self.model_dag.nodes[node_id]["instantiated_class"] = {
                    "key": tag,
                    "object": cls_obj,
                }

            # must have children that would have been instantiated
            else:
                successors = list(self.model_dag.successors(node_id))
                num_successors = len(successors)
                different_tags = set(
                    self.model_dag.nodes[child]["tag"] for child in successors
                )
                num_different_tags = len(different_tags)

                # They are a lst of embedders
                # They all have the same section name
                # but different classes
                if num_successors > 1 and num_different_tags == 1:
                    # pass through concat embedders
                    embedders_ = []
                    unique_tag = different_tags.pop()
                    for successor in successors:
                        node_data = self.model_dag.nodes[successor]
                        embedder = node_data["instantiated_class"]["object"]
                        embedders_.append(embedder)

                    embedder = ConcatEmbedders(embedders=embedders_)

                    # instantiate the current node here
                    class_args[unique_tag] = embedder
                    cls_obj = create_class(
                        classname=classname,
                        module_name=ClassNursery.class_nursery[classname],
                    )
                    cls_obj = cls_obj(**class_args)
                    self.model_dag.nodes[node_id]["instantiated_class"] = {
                        "key": current_node_tag,
                        "object": cls_obj,
                    }

                # use their tags separately as attributes of the parent node
                else:
                    for successor in successors:
                        successor_node_data = self.model_dag.nodes[successor]
                        instantiated_class_data = successor_node_data[
                            "instantiated_class"
                        ]
                        successor_key = instantiated_class_data["key"]
                        successor_obj = instantiated_class_data["object"]
                        class_args[successor_key] = successor_obj

                    cls_obj = create_class(
                        classname=classname,
                        module_name=ClassNursery.class_nursery[classname],
                    )
                    cls_obj = cls_obj(**class_args)
                    self.model_dag.nodes[node_id]["instantiated_class"] = {
                        "key": current_node_tag,
                        "object": cls_obj,
                    }

        return self.model_dag.nodes[root_nodename]["instantiated_class"]["object"]

    def parse_model_section(self):
        """ Parses the Model section of the toml file

            Returns
            -------
            nn.Module
                A torch module representing the model
            """
        model_section = self.doc.get("model")
        if model_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have model section."
                f"Please provide a model section to construct the model"
            )
        self.model_section = model_section
        with self.msg_printer.loading("Loading Model from file"):
            self._form_dag(section_name="model", section=model_section, parent=None)
            # it has to be a DAG (no cycles please!)
            assert nx.dag.is_directed_acyclic_graph(self.model_dag)

            # now we can instantiate the model by parsing in a bottom up manner
            model = self._instantiate_model_using_dag()
        self.msg_printer.good("Finished Loading Model")
        return model

    def parse_engine_section(self):
        """ Parses the engine section of the TOML file

        Returns
        -------
        Engine
            Object of the Engine class

        """
        engine_section = self.doc.get("engine")
        if engine_section is None:
            raise TOMLConfigurationError(
                f"{self.toml_filename} does not have an engine section"
            )
        self.engine_section = engine_section
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

        # parse any other arguments for metric
        metric_args = {"datasets_manager": self.datasets_manager}
        for key, value in metric_section.items():
            if key == "class":
                pass
            else:
                metric_args[key] = value

        metric_cls = create_class(
            module_name=ClassNursery.class_nursery[metric_classname],
            classname=metric_classname,
        )
        train_metric = metric_cls(**metric_args)
        dev_metric = metric_cls(**metric_args)
        test_metric = metric_cls(**metric_args)
        engine_args["train_metric"] = train_metric
        engine_args["validation_metric"] = dev_metric
        engine_args["test_metric"] = test_metric
        engine_args["datasets_manager"] = self.datasets_manager
        engine_args["model"] = self.model
        engine_args["experiment_name"] = self.experiment_name
        engine_args["experiment_hyperparams"] = self.doc

        engine_module = ClassNursery.class_nursery["Engine"]
        engine_classname = "Engine"
        engine_cls = create_class(classname=engine_classname, module_name=engine_module)
        engine = engine_cls(**engine_args)
        return engine

    def run(self):
        self.parse()
        self.engine.run()
