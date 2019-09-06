# ![sciwing logo]( https://sciwing.s3.amazonaws.com/sciwing.png)
A Modern Toolkit for Scientific Document Processing from WING-NUS

[![Build Status](https://travis-ci.com/abhinavkashyap/sciwing.svg?token=AShdNBksk5K9Pxg45w3H&branch=master)](https://travis-ci.com/abhinavkashyap/sciwing) ![Open Issues](https://img.shields.io/github/issues/abhinavkashyap/sciwing) ![Last Commit](https://img.shields.io/github/last-commit/abhinavkashyap/sciwing) [![Updates](https://pyup.io/repos/github/abhinavkashyap/sciwing/shield.svg)](https://pyup.io/repos/github/abhinavkashyap/sciwing/) ![](https://img.shields.io/badge/contributions-welcome-success)



SciWING is a modern framework for Scientific Document Processing from WING-NUS.  It is aimed at facilitating scientific document processing. SciWING believes in modularity and easy to use interfaces for fast prototype development and experimentation in research. It is also packed with many pre-trained models for fundamental tasks in Scientific Document Processing for practitioners.



## Installation 

You can install SciWING from pip 

```zsh
pip install sciwing
```



## Want to Develop?? ![](http://img.shields.io/badge/contributions-welcome-success)

SciWING requires Python 3.7, We recommend that you install `pyenv`. 

Instructions to install pyenv are available  [here](https://github.com/pyenv/pyenv). If you have problems installing python 3.7 on your machine, make sure you have installed all the packages mentioned [here](https://github.com/pyenv/pyenv/wiki/common-build-problems)

1. **Clone from git** 

   https://github.com/abhinavkashyap/sciwing.git

2. `cd sciwing`

3. **Install all the requirements** 

   `pip install -r requirements.txt`

4. **Download spacy models** 

   `python -m spacy download en`

5. **Install the package locally**

   `pip install -e .`

6. **Create directories where sciwing stores embeddings and experiment results**

   `sciwing develop makedirs`

   `sciwing develop download`

   This will take some time to download all the data and embeddings required for development  

   Sip some :coffee:. Come back later 

7. **Run Tests**

   SciWING uses `pytest` for testing. You can use the following command to run tests 

   `pytest tests -n auto --dist=loadfile`

   The test suite is huge and again, it will take some time to run. We will put efforts to reduce the test time in the next iterations.



