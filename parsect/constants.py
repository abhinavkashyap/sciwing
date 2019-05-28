import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


PATHS = dict(
    DATA_DIR=os.path.join(ROOT_DIR, "data"),
    OUTPUT_DIR=os.path.join(ROOT_DIR, "outputs"),
    CONFIGS_DIR=os.path.join(ROOT_DIR, "configs"),
    REPORTS_DIR=os.path.join(ROOT_DIR, "reports"),
)

FILES = dict(
    SECT_LABEL_FILE=os.path.join(ROOT_DIR, "data", "sectLabel.train.data"),
    ELMO_OPTIONS_FILE="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                      "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    ELMO_WEIGHTS_FILE="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                      "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
)
