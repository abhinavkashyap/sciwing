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



## Simple Example 

Example of a model that concatenates a vanilla word embedding and Elmo embedding and then encodes it using a `LSTM2Vec` encoder before finally passing it through a linear layer for classification.



```python
from sciwing.modules.embedders import BowElmoEmbedder
from sciwing.modules.embedders import VanillaEmbedder 
from sciwing.modules.embedders import ConcatEmbedders

from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder 

# initialize a elmo_embedder
elmo_embedder = BowElmoEmbedder()
ELMO_EMBEDDING_DIMENSION = 1024

# Get word embeddings as PyTorch tensors for all the words in the vocab
embedding = dataset.word_vocab.load_embedding()
# initialize a normal embedder with the word embedding 
# EMBEDDING_DIM is the embedding dimension for the word vectors
vanilla_embedder = VanillaEmbedder(embedding=embedding, embedding_dim=EMBEDDING_DIM)

# concatenate the vanilla embedding and the elmo embedding to get a new embedding
final_embedder = ConcatEmbedders([vanilla_embedder, elmo_embedder])
FINAL_EMBEDDING_DIM = EMBEDDING_DIM + ELMO_EMBEDDING_DIMENSION

# instantiate a LSTM2VecEncoder that encodes a sentence to a single vector
encoder = LSTM2VecEncoder(
  emb_dim= FINAL_EMBEDDING_DIM,
  embedder=final_embedder, 
  hidden_dimension=HIDDEN_DIM  
)

# Instantiate a linear classification layer that takes in an encoder and the dimension of the encoding and the number of classes
model = SimpleClassifier(
  encoder=encoder,
  encoding_dim=HIDDEN_DIM,
  num_classes=NUM_CLASSES
)

```



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

