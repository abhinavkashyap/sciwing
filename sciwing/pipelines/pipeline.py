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
        self.ents: Dict[str, Any] = {}

    def __call__(self, doc_name: pathlib.Path):
        if "sections" not in self.disable:
            abstract = self.task_obj_mapping["sections"].extract_abstract_for_file(
                doc_name
            )
            self.ents["abstract"] = abstract

        return self.ents

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
    ents = pdf_pipeline("/Users/abhinav/Downloads/sciwing_arxiv.pdf")
    print(ents)
