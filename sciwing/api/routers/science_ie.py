from fastapi import APIRouter
from sciwing.models.science_ie import ScienceIE

router = APIRouter()

science_ie_model = ScienceIE()


@router.get("/science_ie/{citation}")
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
    predictions = science_ie_model.predict_for_text(citation)
    return {"tags": predictions, "text_tokens": citation.split()}
