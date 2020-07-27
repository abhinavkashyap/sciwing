# ![sciwing logo]( https://sciwing.s3.amazonaws.com/sciwing.png)
A Modern Toolkit for Scientific Document Processing from [WING-NUS](https://wing.comp.nus.edu.sg/). You can find our technical report  here: https://arxiv.org/abs/2004.03807. 

**Note** The previous demo was available at [bit.ly/sciwing-demo](https://bit.ly/sciwing-demo). Due to unavoidable circumstances, it has been moved to [rebrand.ly/sciwing-demo](https://rebrand.ly/sciwing-demo). 

[![Build Status](https://travis-ci.com/abhinavkashyap/sciwing.svg?token=AShdNBksk5K9Pxg45w3H&branch=master)](https://travis-ci.com/abhinavkashyap/sciwing) [![Documentation Status](https://readthedocs.org/projects/sciwing/badge/?version=latest)](https://sciwing.readthedocs.io/en/latest/?badge=latest) ![Open Issues](https://img.shields.io/github/issues/abhinavkashyap/sciwing) ![Last Commit](https://img.shields.io/github/last-commit/abhinavkashyap/sciwing) [![Updates](https://pyup.io/repos/github/abhinavkashyap/sciwing/shield.svg)](https://pyup.io/repos/github/abhinavkashyap/sciwing/) ![](https://img.shields.io/badge/contributions-welcome-success)



SciWING is a modern framework from WING-NUS to facilitate Scientific Document Processing.  It is built on PyTorch and believes in modularity from ground up and easy to use interface. SciWING includes many pre-trained models for fundamental tasks in Scientific Document Processing for practitioners. It has the following advantages:

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

| Task                           | Dataset        | SciWING model                          | SciWING               | Previous Best                                                |
| ------------------------------ | -------------- | -------------------------------------- | --------------------- | ------------------------------------------------------------ |
| Logical Structure Recovery     | SectLabel      | BiLSTM + Elmo Embeddings               | 73.2 (Macro F-score)  | -                                                            |
| Header Normalisation           | SectLabel      | Bag of Words Elmo                      | 93.52 (Macro F-Score) | -                                                            |
| Citation String Parsing        | Neural Parscit | Bi-LSTM-CRF + GloVe + Elmo + Char-LSTM | 88.44 (Macro F-Score) | 90.45 [Prasad et al](https://dl.acm.org/doi/10.5555/3288541.3288551)(not comparable) |
| Citation Intent Classification | SciCite        | Bi-LSTM + Elmo                         | 82.16 (Fscore)        | 82.6 [Cohan et al](https://arxiv.org/pdf/1904.01608.pdf) (without multi-task learning) |
| I2b2 NER                       | I2B2           | Bi-LSTM + Elmo                         | 85.83 (Macro FScore)  | 86.23  [Boukkouri et al](https://www.aclweb.org/anthology/P19-2041/) |
| BC5CDR - NER (Upcoming)        | -              | -                                      | -                     | -                                                            |

   

## Simple Example 

### Using Citation String Parsing 

```python
from sciwing.models.neural_parscit import NeuralParscit 

# instantiate an object 
neural_parscit = NeuralParscit()

# predict on a citation 
neural_parscit.predict_for_text("Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")

# if you have a file of citations with one citation per line 
neural_parscit.predict_for_file("/path/to/filename")
```

Here is the output of the above example:

![Neural Parscit Output](https://parsect-models.s3-ap-southeast-1.amazonaws.com/neural_parscit_output.png)

### Using Citation Intent Classification 

````python
from sciwing.models.citation_intent_clf import CitationIntentClassification 

# instantiate an object 
citation_intent_clf = CitationIntentClassification()

# predict the intention of the citation 
citation_intent_clf.predict_for_text("Abu-Jbara et al. (2013) relied on lexical,structural, and syntactic features and a linear SVMfor classification.")
````



## Running API services 

The APIs are built using [Fast API](https://github.com/tiangolo/fastapi). We have APIs for citation string parsing, citation intent classification and many other models. There are more APIs on the way. To run the APIs navigate into the `api` folder of this repository and run 

```bash
uvicorn api:app --reload
```



## Running the Demos 

The demos are built using [Streamlit](www.streamlit.io). The Demos make use of the APIs. Please make sure that the APIs are running before the demos can be started. Navigate to the app folder and run the demo using streamlit (Installed along with the package). For example, this command runs all the demos. 



**Note:** The demos download the models and the embeddings if already not downloaded and running the first time on your local machine might take time and memory. We have tested this on a 16GB MacBook Pro and works well. All the demos run on CPU for now and does not make use of any GPU, even when present.

````bash
streamlit run all_apps.py
````



## Contributing ![](http://img.shields.io/badge/contributions-welcome-success)

Thank you for your interest in contributing. You can directly email the author at (email omitted for submission purposes). We will be happy to help.


[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/0)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/0)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/1)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/1)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/2)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/2)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/3)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/3)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/4)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/4)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/5)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/5)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/6)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/6)[![](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/images/7)](https://sourcerer.io/fame/abhinavkashyap/abhinavkashyap/sciwing/links/7)

If you want to get involved in the development we recommend that you install SciWING on a local machine using the instructions below. All our classes and methods are documented and hope you can find your way around it.



## Instructions to install SciWING locally

SciWING requires Python 3.7, We recommend that you install `pyenv`. 

Instructions to install pyenv are available  [here](https://github.com/pyenv/pyenv). If you have problems installing python 3.7 on your machine, make sure to check out their common build problems site  [here](https://github.com/pyenv/pyenv/wiki/common-build-problems) and install all dependencies.

1. **Clone from git** 

   `git clone https://github.com/abhinavkashyap/sciwing.git`

   

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

