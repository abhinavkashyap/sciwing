from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict, Any
from sciwing.models.sectlabel import SectLabel
import pathlib


class Pipeline(metaclass=ABCMeta):
    """ All the different pipelines that we will implement will inherit from this
    Abstract class.

    """

    def __init__(self, disable: Tuple = (), **kwargs):
        self.disable = disable
        self.kwargs = kwargs
        self.task_model_mapping = {"sections": {"classname": SectLabel}}

        self.task_obj_mapping = self._instantiate_models()

    def _instantiate_models(self):
        task_obj_mapping = {}
        for task, task_info in self.task_model_mapping.items():
            if task not in self.disable:
                task_class = task_info["classname"]
                # instantiate the class with the keyword arguments
                task_obj = task_class(**self.kwargs)
                task_obj_mapping[task] = task_obj

        return task_obj_mapping


class PdfPipeline(Pipeline):
    def __init__(self, disable: Tuple = ()):
        super(PdfPipeline, self).__init__(disable=disable)
        self.doc: Dict[str, Any] = {}

    def __call__(self, doc_name: pathlib.Path):
        ents = {}
        if "sections" not in self.disable:
            all_info = self.task_obj_mapping["sections"].extract_all_info(
                pdf_filename=doc_name
            )
            ents["abstract"] = all_info["abstract"]
            ents["section_headers"] = all_info["section_headers"]

        self.doc["ents"] = ents

        return self.doc

    def __iter__(self):
        pass


def pipeline(name="pdf_pipeline", disable: Tuple = ()):
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
    pdf_pipeline = pipeline("pdf_pipeline")
    doc = pdf_pipeline("/Users/abhinav/Downloads/sciwing_arxiv.pdf")
    print(doc["ents"])
