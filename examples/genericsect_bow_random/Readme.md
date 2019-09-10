## Generic Section Classifier with Bag of words encoder and random initial word embeddings  

Generic Section Classifier helps in normalizing the different section headers 
to common names. For example "previous work" and "state of the art" can 
be mapped to "related work".
 [The Original paper](https://www.comp.nus.edu.sg/~kanmy/papers/ijdls-SectLabel.pdf)

This example shows how to perform Generic Section Classifier using SciWING.
Here the Bag of Words Encoder is used along with embeddings that are initialized 
randomly and learnt automatically. It also shows how one can change the embedding easily 
to any of the glove embeddings available as part of SciWING.

### Download the data 

``sciwing download data --task genericsect``

Downloads the sectlabel data

## Writing your own python client to perform classification 
The file `genericsect_bow_random.py` shows how to write the client file to run the classifier. 

To run training you can 
`sh genericsect_bow_random.sh`

## Running from a toml file 
The experiment configuration can also be specified using a TOML file. Take a look 
at `genericsect_bow_random.toml` file. 

#### To run training, validation and testing, run 

`sciwing run genericsect_bow_random.toml`

#### To check results after running training using best model 

``sciwing test genericsect_bow_random.toml``

### Running with Glove embeddings 
Sciwing comes with glove embeddings. It is easy to change the 
embeddings to glove 

We illustrate this using the file ``genericsect_bow_glove.toml``
You can see the embedding type has been changed to GloVe 

As before to run training, testing and validation run 
``sciwing run genericsect_bow_glove.toml`` 

