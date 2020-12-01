Tutorials
==========

If you have not installed SciWING on your local machine yet, head over to our
:doc:`installation_getting_started` section first. Here we are going to provide more indepth tutorials
on accessing the pre-trained models, introspecting them, building the models step by step and others.


Examples
------------------
If you would like to see examples of how SciWING is used for training models for different tasks,
the python code for various tasks in SciWING are given in the examples_ folder of our Github Repo.
The instructions to run the code for examples are provided within every example.


.. _examples: https://github.com/abhinavkashyap/sciwing/tree/master/examples




Pre-trained Models
-------------------
.. note::
    If this is your first time use of the package, it takes time to download the pre-trained models.
    Subsequent access to the models are going to be faster.



Neural Parscit
^^^^^^^^^^^^^^^^
Neural Parscit is a citation parsing model. A citation string contains information like the author,
the title of the publication, the conference/journal the year of publication etc.
Neural Parscit extracts such information from references.

.. code-block:: Python

    >> from sciwing.models.neural_parscit import NeuralParscit

    # predict for a citation
    >> neural_parscit = NeuralParscit()

    # Predict on a reference string
    >> neural_parscit.predict_for_text("Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")

    # Predict on a file - The file should contain one referece for string
    >> neural_parascit.predict_for_file("/path/to/file")


Citation Intent Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Identify the intent behind citing another scholarly document helps in fine-grain analysis of documents.
Some citations refer to the methodology in another document, some citations may refer to other works
for background knowledge and some might compare and contrast their methods with another work. Citation
Intent Classification models classify such intents.

.. code-block:: Python

    >> from sciwing.models.citation_intent_clf import CitationIntentClassification

    # instantiate an object
    >> citation_intent_clf = CitationIntentClassification()

    # predict the intention of the citation
    >> citation_intent_clf.predict_for_text("Abu-Jbara et al. (2013) relied on lexical,structural, and syntactic features and a linear SVMfor classification.")


I2B2 Clinical Notes Tagging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clinical Natural Language Processing helps in identifying salient information from clinical notes.
Here, we have trained a neural network model on the **i2b2: Informatics for Integrating Biology and
the Bedside dataset**.This dataset has manual annotation for the **problems** identified, the **treatments**
and **tests** suggested.


.. code-block:: Python

    >> from sciwing.models.i2b2 import I2B2NER

    >> i2b2ner = I2B2NER()

    >> i2b2ner.predict_for_text("Chest x - ray showed no evidency of cardiomegaly")

Extracting Abstracts
^^^^^^^^^^^^^^^^^^^^^^^
You can extract the abstracts from pdf files or abstracts from a folder containing pdf files.

.. code-block:: Python

    >> from sciwing.models.sectlabel import SectLabel

    >> sectlabel = SectLabel()

    # extract abstract for file
    >> sectlabel.extract_abstract_for_file("/path/to/pdf/file")

    # extract abstract for all the files in the folder
    >> sectlabel.extract_abstract_for_folder("/path/to/folder")



Identifying Different Logical Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Identifying different logical sections of the model is a fundamental task in scientific document processing.
The ``SectLabel`` model of SciWING is used to obtain information about different sections of a research
article.

``SectLabel`` can label every line of the document to one of many different labels like
``title``, ``author``, ``bodyText`` etc. which can then be used for many other down-stream
applications.

.. code-block:: Python

    >> from sciwing.models.sectlabel import SectLabel

    >> sectlabel = SectLabel()

    # label all the lines in a document
    >> sectlabel.predict_for_file("/path/to/pdf")

You can also get the ``abstract``, ``section headers`` and the embedded refernces in the document
using the same model as follows


.. code-block:: Python

    >> from sciwing.models.sectlabel import SectLabel

    >> sectlabel = SectLabel()

    >> sectlabel.predict_for_file("/path/to/pdf")

    >> info = sectlabel.extract_all_info("/path/to/pdf")

    >> abstract = info["abstract"]

    >> section_headers = info["section_headers"]

    >> references = info["references]



Normalising Section Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Different research paper use different section headers. However, in order to identify the logical flow
of a research paper, it would be helpful, if we could normalize the different section headers to a
pre-defined set of headers. This model helps in performing such classifications.

.. code-block:: Python

    >> from sciwing.models.generic_sect import GenericSect

    >> generic_sect = GenericSect()

    >> generic_sect.predict_for_text("experiments and results")
    ## evaluation


Interacting with Models
---------------------------------
SciWING allows you to interact with pre-trained models even without writing code. We can interact with
all the pre-trained models using command line application. Upon installation, the command
``sciwing`` is available to the users. One of the sub-commands is the interact command. Let us
see an example

.. code-block:: bash

    sciwing interact neural-parscit

This will run the inference of the best model on test data and prepare the model for interaction.

.. note::
    The inference time again depends on whether you have a CPU or GPU. By default, we assume
    that you are running the model on a CPU.

.. code-block:: bash

  1. See-Confusion-Matrix
  2. See-examples-of-Classifications
  3. See-prf-table
  4. Enter text

1.The first option shows confusion matrix for different classes of `Neural ParsCit`.

2.The second option shows examples where one class is misclassified as the other. For eg., enter
``4 5`` to show examples where some tags belonging to class `4` is misclassified as ``5``

3.The Precision Recall and F-measure for the test dataset is shown along with the macro and micro F-scores.

4.You can enter a reference string and see the results.


PDF Pipelines
-----------------------
.. note::
    **Under Construction**: This will allow you to provide path to a PDF file and extract all the
    information with respect to the pdf file. The information includes abstract, title, author,
    section headers, normalized section headers, embedded references, parses of the references etc.




