from fastapi import APIRouter
from sciwing.models.i2b2 import I2B2NER

router = APIRouter()

i2b2_ner_model = None


@router.get("/i2b2/{text}")
def return_tags(text: str):
    """ Tags the text that you send according to i2b2 model with
    ``problem, treatment and tests``

    Parameters
    ----------
    text: str
        The text to be tagged

    Returns
    -------
    JSON
        ``{tags: Predicted tags, text_tokens: Tokens in the text }``

    """
    global i2b2_ner_model
    if i2b2_ner_model is None:
        i2b2_ner_model = I2B2NER()

    predictions = i2b2_ner_model.predict_for_text(text)
    return {"tags": predictions, "text_tokens": text.split()}
