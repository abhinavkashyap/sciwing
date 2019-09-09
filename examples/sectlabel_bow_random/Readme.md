## Sect Label with Bag of words encoder and random initial word embeddings  

SectLabel is aimed at performing logical section classification. 23 different classes are 
considered for classification. More information about SectLabel Classification 
can be obtained from [The Original paper](https://www.comp.nus.edu.sg/~kanmy/papers/ijdls-SectLabel.pdf)

This example shows how to perform Logical Section Classifier using SciWING.
Here the Bag of Words Encoder is used along with embeddings that are initialized 
randomly and learnt automatically. It also shows how one can change the embedding easily 
to any of the glove embeddings available as part of SciWING.

### Download the data 

``sciwing download data sectlabel``

Downloads the sectlabel data

## Writing your own python client to perform classification 
The file `sectlabel_bow_random.py` shows how to write the client file to run the classifier. 

To run training you can 
`sh sectlabel_bow_random.sh`

## Running from a toml file 
The experiment configuration can also be specified using a TOML file. Take a look 
at `sectlabel_bow_random.toml` file. 

#### To run training, validation and testing, run 

`sciwing run sectlabel_bow_random.toml`

#### To check results after running training using best model 

``sciwing test sectlabel_bow_random.toml``