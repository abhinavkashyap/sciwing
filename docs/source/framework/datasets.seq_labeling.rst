*********************
Sequential Labeling
*********************
Datasets for Sequential Labeling. In Sequential Labeling, the entire sequence is labeled.
A typical example is to do Named Entity Recognition where given a sentence
spans of text are labelled as an organization or a place amongst other.
In Scientific Domain, Sequential Labeling is often used in different scenarios. Reference
String parsing is one such scenario.


Parscit Dataset
===================
 Parscit is a system by WING-NUS, that enables reference string parsing. The dataset is available here_.


.. _here: https://raw.githubusercontent.com/knmnyn/ParsCit/master/crfpp/traindata/cora.train
.. autoclass:: parsect.datasets.seq_labeling.parscit_dataset.ParscitDataset
    :members:


Science IE Dataset
==================
.. autoclass:: parsect.datasets.seq_labeling.science_ie_dataset.ScienceIEDataset
    :members:
