ScienceIETagger 
-----------------

Science IE is a SemEval Task that involves keyphrase extraction 
from scientific text. The tagging involves tagging whether a phrase is PROCESS - 
phrases relation to some scientific model, TASK - phrases denoting the application or goal 
of the problem and MATERIAL - phrases that identify the different resources identified in the paper

The keyphrase identification is a sequnce tagging probem.
In this example we use the ScienceIETagger that is a bidirectional LSTM model with CRF. 
At the embedding layer the character embedding and the word level embedding is combined.

### Download the data 

``sciwing download data --task scienceie``

Downloads the science ie data 

## Writing your own python client to perform classification 
The file `science_ie.py` shows how to write the client file for ScienceIE Tagging

To run training you can 
`sh science_ie.sh`

## Running from a toml file 
The experiment configuration can also be specified using a TOML file. Take a look 
at `science_ie.toml` file. 

#### To run training, validation and testing, run 

`sciwing run science_ie.toml`

#### To check results after running training using best model 

``sciwing test science_ie.toml``