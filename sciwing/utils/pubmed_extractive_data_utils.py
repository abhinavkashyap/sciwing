from typing import List, Dict
import wasabi
from collections import OrderedDict
from tqdm import tqdm
import json
from pathlib import Path, PurePath


def write_extractive_to_sciwing_text_clf(
    extractive_data_dir: str, data_group: str, out_filename: str
):
    """
    The preprocessed extractive summarization dataset contains 3 folders:
        human-abstracts (ground truth),
        inputs (document id, original sentences, tokenized sentences),
        labels (document id, labels of each sentence in the document indicating whether this sentence should be included
                in the summary).
    Each folder has 3 sub-folders: train, test, val. Each json file under the sub-folder contains one document.

    Parameters
    ----------
    extractive_data_dir : str
        The directory where all the data file are stored

    data_group:
        Choose from train, dev, and test. Dev is corresponding to the val folder in the input data

    out_filename:
        The output filename where the extractive summarization dataset is stored

    Returns
    -------
    None
    """
    printer = wasabi.Printer()
    document = []
    input_data_human_abstract_dir = Path(
        extractive_data_dir, "human-abstracts", data_group
    )
    input_data_inputs_dir = Path(extractive_data_dir, "inputs", data_group)
    input_data_labels_dir = Path(extractive_data_dir, "labels", data_group)
    filename_list = [f.stem for f in input_data_human_abstract_dir.iterdir()]

    with printer.loading(f"Writing f{out_filename}"):
        for filename in filename_list:
            ha_filename = input_data_human_abstract_dir.joinpath(f"{filename}.text")
            input_filename = input_data_inputs_dir.joinpath(f"{filename}.json")
            label_filename = input_data_labels_dir.joinpath(f"{filename}.json")

            with open(ha_filename, "r") as fp:
                abstract_str = fp.read().strip()
                abstract_str = abstract_str.strip().replace("\n", " ")

            with open(input_filename, "r") as fp:
                input_dict = json.load(fp)
                input_str = [
                    sent["text"].strip().remove("\n") for sent in input_dict["inputs"]
                ]

            with open(label_filename, "r") as fp:
                label_dict = json.load(fp)
                label_str = []


if __name__ == "__main__":
    import sciwing.constants as constants
    import pathlib

    PATHS = constants.PATHS
    DATA_DIR = PATHS["DATA_DIR"]

    data_dir = pathlib.Path(DATA_DIR)
    pubmed_extractive_data_dir = data_dir.joinpath("pubmed")

    pubmed_extractive_train_filename = data_dir.joinpath("pubmed_extractive.train")
    pubmed_extractive_dev_filename = data_dir.joinpath("pubmed_extractive.dev")
    pubmed_extractive_test_filename = data_dir.joinpath("pubmed_extractive.test")
