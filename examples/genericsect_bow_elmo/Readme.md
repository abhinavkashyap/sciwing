## Generic Section Classifier with Bag of words encoder and ELMO representations  

Generic Section Classifier helps in normalizing the different section headers 
to common names. For example "previous work" and "state of the art" can 
be mapped to "related work".
 [The Original paper](https://www.comp.nus.edu.sg/~kanmy/papers/ijdls-SectLabel.pdf)

This example shows how to perform Generic Section Classifier using SciWING.
Here the Bag of Words Encoder is used along with Elmo Embeddings and then 
classified using a linear classifier 


### Download the data 

``sciwing download data --task genericsect``

Downloads the sectlabel data

## Writing your own python client to perform classification 
The file `genericsect_bow_elmo.py` shows how to write the client file to run the classifier. 

To run training you can 
`sh genericsect_bow_elmo.sh`

## Running from a toml file 
The experiment configuration can also be specified using a TOML file. Take a look 
at `genericsect_bow_elmo.toml` file. 

#### To run training, validation and testing, run 

`sciwing run genericsect_bow_elmo.toml`

#### To check results after running training using best model 

``sciwing test genericsect_bow_elmo.toml``


