# ![sciwing logo]( https://sciwing.s3.amazonaws.com/sciwing.png)
A Modern Toolkit for Scientific Document Processing from [WING-NUS](https://wing.comp.nus.edu.sg/)

[![Build Status](https://travis-ci.com/abhinavkashyap/sciwing.svg?token=AShdNBksk5K9Pxg45w3H&branch=master)](https://travis-ci.com/abhinavkashyap/sciwing) ![Open Issues](https://img.shields.io/github/issues/abhinavkashyap/sciwing) ![Last Commit](https://img.shields.io/github/last-commit/abhinavkashyap/sciwing) [![Updates](https://pyup.io/repos/github/abhinavkashyap/sciwing/shield.svg)](https://pyup.io/repos/github/abhinavkashyap/sciwing/) ![](https://img.shields.io/badge/contributions-welcome-success)



SciWING is a modern framework from WING-NUS to facilitate Scientific Document Processing.  It is built on PyTorch and believes in modularity from ground up and easy to use interface. SciWING includes many pre-trained models for fundamental tasks in Scientific Document Processing for practitioners. It has the following advantages

- **Modularity**  - The framework embraces modularity from ground-up. **SciWING** helps in creating new models by combining multiple re-usable modules. You can combine different modules and experiment with new approaches in an easy manner 

- ***Pre-trained Models*** - SciWING has many pre-trained models for fundamental tasks like Logical Section Classifier for scientific documents, Citation string Parsing (Take a look at some of the other project related to station parsing [Parscit](https://github.com/WING-NUS/ParsCit), [Neural Parscit](https://github.com/WING-NUS/Neural-ParsCit). Easy access to pre-trained models are made available through web APIs.

- ***Run from Config File***- SciWING enables you to declare datasets, models and experiment hyper-params in a [TOML](https://github.com/toml-lang/toml) file. The models declared in a TOML file have a one-one correspondence with their respective class declaration in a python file. SciWING parses the model to a Directed Acyclic Graph and instantiates the model using the DAG's topological ordering.

- **Extensible** - SciWING enables easy addition of new datasets and provides command line tools for it. It enables addition of custom modules which are PyTorch modules.

  



## Installation 

You can install SciWING from pip. We recommend using a virtual environment to install the package. 

```zsh
pip install sciwing
```



## Tasks 

These are some of the tasks included in SciWING and their performance metrics 

| Task                               | Dataset        | SciWING model                          | SciWING | Previous Best                      |
| ---------------------------------- | -------------- | -------------------------------------- | ------- | ---------------------------------- |
| Logical Structure Recovery         | SectLabel      |                                        |         | -                                  |
| Header Normalisation               | SectLabel      |                                        |         | -                                  |
| Citation String Parsing            | Neural Parscit | Bi-LSTM-CRF + GloVe + Elmo + Char-LSTM | 88.44   | 90.45(not comparable)              |
| Citation Intent Classification     | SciCite        | Bi-LSTM + Elmo                         | 82.16   | 82.6 (without multi-task learning) |
| Biomedical NER - BC5CDR (Upcoming) | -              | -                                      | -       | -                                  |
| I2b2 NER (Upcoming)                | -              | -                                      | -       | -                                  |



## Simple Example 

### Using Citation String Parsing 

```python
from sciwing.models.neural_parscit import NeuralParscit 

# instantiate an object 
neural_parscit = NeuralParscit()

# predict on a citation 
neural_parscit.predict_for_text("")

# if you have a file of citations with one citation per line 
neural_parscit.predict_for_file("/path/to/filename")
```



### Using Citation Intent Classification 

````python
from sciwing.models.citation_intent_clf import CitationIntentClassification 

# instantiate an object 
citation_intent_clf = CitationIntentClassification()
````



## Running API services 



## Running the Demos 







## Contributing ![](http://img.shields.io/badge/contributions-welcome-success)

Thank you for your interest in contributing. You can directly email the author at (email omitted for submission purposes). We will be happy to help.



If you want to get involved in the development we recommend that you install SciWING on a local machine using the instructions below. All our classes and methods are documented and hope you can find your way around it.



## Instructions to install SciWING locally

SciWING requires Python 3.7, We recommend that you install `pyenv`. 

Instructions to install pyenv are available  [here](https://github.com/pyenv/pyenv). If you have problems installing python 3.7 on your machine, make sure to check out their common build problems site  [here](https://github.com/pyenv/pyenv/wiki/common-build-problems) and install all dependencies.

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

