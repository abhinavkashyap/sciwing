from fastapi import APIRouter
from sciwing.models.neural_parscit import NeuralParscit

router = APIRouter()

parscit_model = None


@router.get("/parscit/{citation}")
def tag_citation_string(citation: str):
    """ End point to tag parse a reference string to their constituent parts.

    Parameters
    ----------
    citation: str
        The reference string to be parsed.

    Returns
    -------
    JSON
        ``{"tags": Predicted tags, "text_tokens": Tokenized citation string}``

    """
    global parscit_model
    if parscit_model == None:
        parscit_model = NeuralParscit()
    predictions = parscit_model.predict_for_text(citation)
    return {"tags": predictions, "text_tokens": citation.split()}
