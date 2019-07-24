Package Reference
********************

Datasets
=========

Classification
###############
Dataset for Text Classification. Text classification typicall takes a sentence and classifies to
different classes.

In this package we have included **SectLabel** dataset, that performs logical
section classification and the **GenSect** classification datasets.

Parsect Dataset
----------------
.. autoclass:: parsect.datasets.classification.parsect_dataset.ParsectDataset
    :members:

GenericSect Dataset
-------------------
.. automodule:: parsect.datasets.classification.generic_sect_dataset
    :members:


Sequential Labeling
####################
Datasets for Sequential Labeling. In Sequential Labeling, the entire sequence is labeled.
A typical example is to do Named Entity Recognition where given a sentence
spans of text are labelled as an organization or a place amongst other.
In Scientific Domain, Sequential Labeling is often used in different scenarios


Parscit Dataset
----------------
.. automodule:: parsect.datasets.seq_labeling.parscit_dataset
    :members:

Science IE Dataset
------------------
.. automodule:: parsect.datasets.seq_labeling.science_ie_dataset
    :members:


Your Dataset Here
#####################
