.. SciWing documentation master file, created by
   sphinx-quickstart on Wed Jul 24 16:27:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SciWing's documentation!
===================================
   .. image:: ./_static/img/logo.png
         :width: 300

   SciWING is a modular and easy to extend framework, that enables easy experimentation
   of modern techniques for Scholarly Document Processing. It enables easy addition of datasets,
   and models and provides tools to easily experiment with them.

   SciWING is a modern framework from WING-NUS to facilitate Scientific Document Processing.
   It is built on PyTorch and believes in modularity from ground up and easy to use interface.
   SciWING includes many pre-trained models for fundamental tasks in Scientific Document
   Processing for practitioners. It has the following advantages

   - **Modularity**  - The framework embraces modularity from ground-up. **SciWING** helps in creating new models by combining multiple re-usable modules. You can combine different modules and experiment with new approaches in an easy manner

   - **Pre-trained Models** -SciWING has many pre-trained models for fundamental tasks like Logical SectionClassifier for scientific documents, Citation string Parsing(Take a look at some of the other project related to station parsing Parscit_, Neural_Parscit_ . Easy access to pre-trained models are made available through web APIs.

   - **Run from Config File**- SciWING enables you to declare datasets, models and experiment hyper-params in a TOML_ file. The models declared in a TOML file have a one-one correspondence with their respective class declaration in a python file. SciWING parses the model to a Directed Acyclic Graph and instantiates the model using the DAG's topological ordering.

   - **Extensible** - SciWING enables easy addition of new datasets and provides command line tools for it. It enables addition of custom modules which are PyTorch modules.

   .. _Parscit: https://github.com/WING-NUS/ParsCit
   .. _Neural_Parscit: https://github.com/WING-NUS/Neural-ParsCit
   .. _TOML: https://github.com/toml-lang/toml


.. toctree::
   :maxdepth: 4

   usage/index
   framework/index

