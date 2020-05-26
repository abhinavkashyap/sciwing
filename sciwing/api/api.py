import sciwing.api.conf as config
from fastapi import FastAPI
from sciwing.api.routers import parscit
from sciwing.api.routers import citation_intent_clf
from sciwing.api.routers import sectlabel
from sciwing.api.routers import parscit, generic_sect
from sciwing.api.routers import science_ie

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome To SciWING API"}


# add the routers to the main app
app.include_router(parscit.router)
app.include_router(citation_intent_clf.router)
app.include_router(sectlabel.router)

app.include_router(science_ie.router)
app.include_router(generic_sect.router)