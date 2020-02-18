import falcon
from typing import Dict, Callable, Any


class ScienceIETaggerResource:
    def __init__(self, model_filepath: str, model_infer_func: Callable):
        """
        Parameters
        ----------
        model_filepath: str
            The path to the directory where the model experiment is stored
        model_infer_func : Callable
            The model infer function is a function that takes in the experiment
            directory and can return the inference client. The inference
            client is an object of infer.seq_label_inference.parscit_inf
        """
        self.model_filepath = model_filepath
        self.model_infer_func = model_infer_func
        self.model_infer_client = None

    def on_get(self, req, resp) -> Dict[str, Any]:
        """ The GET method takes ``text`` in the query parameter
        and return three tagged strings. The three tags are ``[Task, Process, Material]``

        Returns
        -------
        Dict[str, Any]
            Returns a json with the following fields
            ``{text: "",task_tags: [], process_tags: [], material_tags: []}``

        """
        if self.model_infer_client is None:
            self.model_infer_client = self.model_infer_func(self.model_filepath)

        science_text = req.get_param("text")

        if science_text is None:
            resp.status = falcon.HTTP_400
            resp.body = f"text is not present in the request"

        else:
            (
                task_tags,
                process_tags,
                material_tags,
            ) = self.model_infer_client.infer_single_sentence(science_text)
            resp.status = falcon.HTTP_200
            resp.media = {
                "text": science_text,
                "task_tags": task_tags,
                "process_tags": process_tags,
                "material_tags": material_tags,
            }
