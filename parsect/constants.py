import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


PATHS = dict(
    DATA_DIR=os.path.join(ROOT_DIR, 'data'),
    OUTPUT_DIR=os.path.join(ROOT_DIR, 'outputs')
)

FILES = dict(
    SECT_LABEL_FILE=os.path.join(ROOT_DIR, 'data', 'sectLabel.train.data')
)