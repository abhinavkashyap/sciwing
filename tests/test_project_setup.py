from parsect.project_setup import project_setup
import pytest
import parsect.constants as constants
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS['DATA_DIR']
OUTPUT_DIR = PATHS['OUTPUT_DIR']


def test_data_dir_exists():
    path = pathlib.Path(DATA_DIR)
    is_dir = path.is_dir()
    assert is_dir


def test_sect_label_file_exists():
    path = pathlib.Path(DATA_DIR, 'sectLabel.train.data')
    is_file = path.is_file()
    assert is_file


def test_embeddings_dir():
    path = pathlib.Path(DATA_DIR, 'embeddings')
    is_dir = path.is_dir()
    assert is_dir


def test_glove_dir():
    path = pathlib.Path(DATA_DIR, 'embeddings', 'glove')
    is_glove_dir = path.is_dir()
    assert is_glove_dir


def test_glove_embeddings_file_exists():
    path = pathlib.Path(DATA_DIR, 'embeddings', 'glove')
    filenames = ['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    for filename in path.iterdir():
        assert str(filename.stem) in filenames


def test_output_dir_exists():
    path = pathlib.Path(OUTPUT_DIR)
    assert path.is_dir()
