*******************
Classification
*******************
Dataset for Text Classification Tasks.
Text classification typicall takes a sentence and classifies to different classes.

Dividing a scholarly document into different logical sections like introduction, related work, conclusion
is an important task. SciWING includes SectLabel_ dataset fromt he WING-NUS group aimed at this task.
A scholarly document also uses the same name for different sections. For example,
methodology, method and experiments all indicate a single logical section.
SciWING also includes GenericSect_ dataset from the WING-NUS group aimed at this task.

.. _SectLabel: https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/sectLabel.train.data
.. _GenericSect: https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/genericSect.train.data

SectLabel Dataset
==================
.. automodule:: parsect.datasets.classification.sectlabel_dataset
    :members:
    :undoc-members:
    :show-inheritance:


GenericSect Dataset
=====================
.. autoclass:: parsect.datasets.classification.generic_sect_dataset.GenericSectDataset
    :members:
    :undoc-members:
    :show-inheritance:
