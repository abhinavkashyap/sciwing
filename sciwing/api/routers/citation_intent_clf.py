from fastapi import APIRouter
from sciwing.models.citation_intent_clf import CitationIntentClassification

router = APIRouter()

citation_intent_clf_model = None


@router.get("/cit_int_clf/{citation}")
def classify_citation_intent(citation: str):
    """ End point to classify a citation intent into ```Background`, `Method`, `Result Comparison```

    Parameters
    ----------
    citation : str
        String containing the citation to another work

    Returns
    -------
    JSON
        ``{"tags": Predicted tag for the citation, "citation": the citation itself}``
    """
    global citation_intent_clf_model
    if citation_intent_clf_model is None:
        citation_intent_clf_model = CitationIntentClassification()
    predictions = citation_intent_clf_model.predict_for_text(citation)
    return {"tags": predictions, "citation": citation}
