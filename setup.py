from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="sciwing",
    version="0.1.0b1",
    packages=find_packages(exclude=("tests",)),
    url="",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern ParseSect and ParseHed projects from WING-NUS",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing_group:main"]},
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        "networkx",
        "wandb",
        "logzero,",
        "falcon_multipart",
        "typing==3.6.6",
        "torch==1.1.0",
        "wasabi==0.2.2",
        "boto3==1.9.211",
        "tqdm==4.32.1",
        "wrapt==1.11.2",
        "stop_words",
        "allennlp==0.8.3",
        "botocore",
        "gensim==3.8.0",
        "pytest==4.5.0",
        "pytest-sugar",
        "pytest-xdist",
        "pytorch_pretrained_bert",
        "spacy==2.1.4",
        "questionary==1.1.1",
        "json_logging",
        "autopep8==1.4.4",
        "pandas==0.24.2",
        "pytorch_crf",
        "colorful==0.5.1",
        "Jinja2==2.10.1",
        "falcon==2.0.0",
        "numpy==1.16.3",
        "click==7.0",
        "toml==0.10.0",
        "requests==2.22.0",
        "scikit_learn==0.21.3",
        "tensorboardX==1.8",
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-autobuild",
        "Deprecated",
    ],
)
