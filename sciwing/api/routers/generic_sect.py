from fastapi import APIRouter
from sciwing.models.generic_sect import GenericSect

router = APIRouter()

generic_sect_model = GenericSect()


@router.get("/generic_sect/{citation}")
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
    predictions = generic_sect_model.predict_for_text(citation)
    return {"tags": predictions, "text_tokens": citation.split()}
