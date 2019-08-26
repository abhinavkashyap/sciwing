import falcon
from typing import Dict, Callable


class ParscitTaggerResource:
    def __init__(self, model_filepath: str, model_infer_func: Callable):
        self.model_filepath = model_filepath
        self.model_infer_func = model_infer_func
        self.model_infer_client = None

    def on_get(self, req, resp) -> Dict[str, str]:
        """ Get the tagged reference string
        Returns
        -------
        Dict[str, str]
            Returns the tagged string
        """
        if self.model_infer_client is None:
            self.model_infer_client = self.model_infer_func(self.model_filepath)

        citation_string = req.get_param("citation")

        if citation_string is None:
            resp.status = falcon.HTTP_400
            resp.body = f"citation not found in request"

        else:
            citation_labels = self.model_infer_client.infer_single_sentence(
                citation_string
            )
            resp.status = falcon.HTTP_200
            resp.media = {"citation": citation_string, "label": citation_labels}
