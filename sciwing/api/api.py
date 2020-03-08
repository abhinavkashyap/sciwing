import sciwing.api.conf as config
from fastapi import FastAPI
from sciwing.api.routers import parscit

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome To SciWING API"}


# add the routers to the main app
app.include_router(parscit.router)
