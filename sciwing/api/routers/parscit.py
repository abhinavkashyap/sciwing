from fastapi import APIRouter
from sciwing.models.neural_parscit import NeuralParscit

router = APIRouter()

parscit_model = NeuralParscit()


@router.get("/parscit/{citation}")
def tag_citation_string(citation: str):
    """ End point to tag a citation

    Parameters
    ----------
    citation: str

    Returns
    -------
    JSON
        Predicted tags for the given citation

    """
    predictions = parscit_model.predict_for_text(citation)
    return {"tags": predictions, "text_tokens": citation.split()}
