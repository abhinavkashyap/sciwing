import sciwing.api.conf as config
from fastapi import FastAPI
from sciwing.api.routers import parscit
from sciwing.api.routers import citation_intent_clf
from sciwing.api.routers import sectlabel

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome To SciWING API"}


# add the routers to the main app
app.include_router(parscit.router)
app.include_router(citation_intent_clf.router)
app.include_router(sectlabel.router)
