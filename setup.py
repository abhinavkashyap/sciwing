from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="sciwing",
    version="0.1.post9",
    packages=find_packages(exclude=("tests",)),
    url="https://github.com/abhinavkashyap/sciwing",
    license="",
    author="abhinav",
    author_email="abhinav@comp.nus.edu.sg",
    description="Modern Scientific Document Processing Framework ",
    entry_points={"console_scripts": ["sciwing=sciwing.commands.sciwing_group:main"]},
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        "streamlit",
        "torch==1.5.0",
        "colorful==0.5.1",
        "pytest==5.4.1",
        "psutil==5.7.0",
        "spacy==2.1.4",
        "boto3",
        "pandas",
        "gensim==3.8.0",
        "Deprecated==1.2.7",
        "networkx==2.2",
        "wasabi==0.2.2",
        "logzero==1.5.0",
        "flair==0.5",
        "tqdm==4.32.1",
        "allennlp==0.8.3",
        "wandb==0.8.29",
        "toml==0.10.0",
        "numpy",
        "pytorch_pretrained_bert==0.6.2",
        "stop_words==2018.7.23",
        "questionary==1.1.1",
        "fastapi==0.65.1",
        "scikit_learn==0.22.2",
        "tensorboardX==2.2",
        "werkzeug==1.0.0",
        "overrides==3.1.0",
        "s3transfer==0.4.0",
        "requests",
        "Sphinx==2.4.4",
        "Jinja2",
        "Flask==1.1.1",
        "Flask-Cors==3.0.8",
        "importlib-metadata==3.7.0",
        "docutils==0.15",
    ],
)
