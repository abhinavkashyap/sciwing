from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Any, List
from sciwing.models.sectlabel import SectLabel
from sciwing.models.neural_parscit import NeuralParscit
from sciwing.models.generic_sect import GenericSect
import pathlib
from collections import defaultdict
import wasabi


class Pipeline(metaclass=ABCMeta):
    """ All the different pipelines that we will implement will inherit from this
    Abstract class.

    """

    def __init__(self, disable: List = [], **kwargs):
        self.disable = disable
        self.kwargs = kwargs

        # maps the different tasks to the respective model names
        # tasks might be associated with multiple models
        # For example, extracting different sections requires sectlabel model, generic
        # section classification model etc
        self.task_model_mapping = {
            "sections": {"classnames": [SectLabel]},
            "reference-string-extract": {"classnames": [NeuralParscit]},
            "normalize-section-headers": {"classnames": [GenericSect]},
        }

        # mapping between the task and all the istantiated models with
        self.task_obj_mapping = self._instantiate_models()

    def _instantiate_models(self):
        task_obj_mapping = defaultdict(list)
        for task, task_info in self.task_model_mapping.items():
            if task not in self.disable:
                task_classes = task_info["classnames"]
                # instantiate the class with the keyword arguments
                for task_class in task_classes:
                    task_obj = task_class(**self.kwargs)
                    task_obj_mapping[task].append(task_obj)

        return task_obj_mapping


class PdfPipeline(Pipeline):
    def __init__(self, disable: List = []):
        super(PdfPipeline, self).__init__(disable=disable)
        self.doc: Dict[str, Any] = {}

    def __call__(self, doc_name: pathlib.Path):
        ents = {}

        # By default we extract sections information
        # which is one time thing
        sections_info = self.task_obj_mapping["sections"][0].extract_all_info(
            pdf_filename=doc_name
        )
        ents["abstract"] = sections_info["abstract"]
        ents["section_headers"] = sections_info["section_headers"]
        ents["references"] = sections_info["references"]

        # reference string parsing
        if "reference-string-extract" not in self.disable:
            parsed_ref_strings = []
            for reference_text in sections_info["references"]:
                parsed_string = self.task_obj_mapping["reference-string-extract"][
                    0
                ].predict_for_text(text=reference_text, show=False)
                parsed_ref_strings.append(parsed_string)

            ents["parsed_reference_strings"] = parsed_ref_strings

        if "normalize-section-headers" not in self.disable:
            normalized_section_headers = []
            for sect_header in sections_info["section_headers"]:
                norm_sect_header = self.task_obj_mapping["normalize-section-headers"][
                    0
                ].predict_for_text(text=sect_header)
                normalized_section_headers.append(norm_sect_header)

            ents["normalized_section_headers"] = normalized_section_headers

        self.doc["ents"] = ents

        return self.doc

    def __iter__(self):
        """ You will be able to iterate over the class and obtain the different entities of the pdf
        document

        Returns
        -------

        """
        for name, entity in self.doc["ents"].items():
            yield name, entity


def pipeline(name="pdf_pipeline", disable: List = []):
    """ Defines a pipeline function
    It just takes in a name and instantiates the pipeline

    Parameters
    ----------
    name : str
        The name of the pipeline that you would want
    disable: Tuple
        You can disable some model loading by passing in a tuple

    """
    if name == "pdf_pipeline":
        pipeline = PdfPipeline(disable=disable)

    return pipeline


if __name__ == "__main__":
    pdf_pipeline = pipeline()
    pdf_path = pathlib.Path("/Users/abhinav/Downloads/sciwing_arxiv.pdf")
    ents = pdf_pipeline(pdf_path)
    for name, ent in ents["ents"].items():
        print(f"name: {name} \n entity: {ent} \n")
