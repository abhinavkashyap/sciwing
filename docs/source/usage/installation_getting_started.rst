Installation and Getting Started
---------------------------------

The first step to use SciWING is to install the package on your local system. Once the package is
installed, you can directly access the functionalities of SciWING. SciWING downloads the
pre-trained models, embeddings and other information that is required to run the models
on-demand basis.

On this page, we provide some basic tutorials on installation of SciWING and basic usage of SciWING.


Installation from Pip
----------------------------
SciWING works with Python 3.7 or later. Currently this is the default and the only way to install
SciWING using Pip, the python package manager. We recommend using
``virtualenv`` which helps in keeping the development environment clean. To install, just run

.. code-block:: Bash

    pip install sciwing

This installs all the dependencies required to run SciWING like ``PyTorch``.

If you want to install sciwing for the current user then you can use

.. code-block:: Bash

    pip install -U sciwing


Building from source
-------------------------------------

    - Clone from git

    .. code-block:: bash

        git clone https://github.com/abhinavkashyap/sciwing.git

    - cd sciwing

    - Install the module in development mode

    .. code-block:: bash

        pip install -e .

    - Download spacy models

    .. code-block:: bash

        python -m spacy download en

    - Create directories where SciWING's data are stored and embeddings/data are downloaded

    .. code-block:: bash

        sciwing develop makedirs
        sciwing develop download

    This step is optional. This will download the data and embeddings required for development.
    If you do not perform this step, then it gets downloadd later upon first request

    - SciWING uses ``pytest`` for testing. You can use the following command to run tests

    .. code-block:: bash

        pytest tests -n auto --dist=loadfile

    The test suite is huge and again, it will take some time to run. We will put efforts to reduce the test time in the next iterations.

Running API Services
---------------------
The APIs are built using FastAPI_. We have APIs for citation string parsing, citation and intent
classification and many other models. To run the APIs navigate into the api folder of this repository and run

.. _FastAPI: https://fastapi.com

.. code-block:: bash

    uvicorn api:app --reload

.. note::
    Navigate to http://localhost:8000/docs to access the SwaggerUI. The UI enables you to try
    the different APIs using a web interface.

Running the Demos
------------------
The demos are built using Streamlit_. The Demos make use of the APIs. Please make sure that the
APIs are running before the demos can be started. Navigate to the app folder and run the demo using
streamlit (Installed along with the package). For example, this command runs all the demos.

.. _Streamlit: streamlit.io

.. note::
 The demos download the models and the embeddings if already not downloaded and running the first time
 on your local machine might take time and memory. We have tested this on a 16GB MacBook Pro and
 works well. All the demos run on CPU for now and does not make use of any GPU, even when present.

.. code-block:: bash

    streamlit run all_apps.py


Accessing Models
--------------------
SciWING comes with many pre-trained scientific documenting processing models, that are easily
accessible using a few lines of Python code. SciWING provides a consistent interface
for all of its models. You can access these models, immediately after installation.
The required model parameters, the embeddings etc are downloaded and initialized.

.. note::
    The first time access of these models takes time, since we need to download them. Allow 60s, for the
    downloads to complete. Future access of the models are faster


Citation String Parsing
^^^^^^^^^^^^^^^^^^^^^^^^^
Neural Parscit is a citation parsing model. A citation string contains information like the author,
the title of the publication, the conference/journal the year of publication etc.
Neural Parscit extracts such information from references.

.. code-block:: Python

    from sciwing.models.neural_parscit import NeuralParscit

    # predict for a citation
    neural_parscit = NeuralParscit()

    # Predict on a reference string
    neural_parscit.predict_for_text("Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")

    # Predict on a file - The file should contain one referece for string
    neural_parascit.predict_for_file("/path/to/file")


