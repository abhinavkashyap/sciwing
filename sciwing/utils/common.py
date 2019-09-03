from typing import Dict, List, Any, Iterable, Iterator, Tuple

import math
import requests
from wasabi import Printer
import zipfile
from sys import stdout
import re
import pathlib
from sklearn.model_selection import KFold
import numpy as np
import sciwing.constants as constants
from itertools import tee
import importlib
from tqdm import tqdm
import tarfile
import psutil

PATHS = constants.PATHS
FILES = constants.FILES
DATA_DIR = PATHS["DATA_DIR"]
CORA_FILE = FILES["CORA_FILE"]
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]


def convert_sectlabel_to_json(filename: str) -> Dict:
    """ Converts the secthead file into more readable json format

    Parameters
    ----------
    filename : str
        The sectlabel file name available at WING-NUS website

    Returns
    -------
    Dict[str, Any]
        text
            The text of the line
        label
            The label of the file
        file_no
            A unique file number
        line_count
            A line count within the file

    """
    file_count = 1
    line_count = 1
    output_json = {"parse_sect": []}
    msg_printer = Printer()

    with open(filename) as fp:
        for line in tqdm(fp, desc="Converting SectLabel File to JSON"):
            line = line.replace("\n", "")

            # if the line is empty then the next line is the beginning of the
            if not line:
                file_count += 1
                continue

            # new file
            fields = line.split()
            line_content = fields[0]  # first column contains the content text
            line_content = line_content.replace(
                "|||", " "
            )  # every word in the line is sepearted by |||
            label = fields[-1]  # the last column contains the field marked
            line_json = {
                "text": line_content,
                "label": label,
                "file_no": file_count,
                "line_count": line_count,
            }
            line_count += 1

            output_json["parse_sect"].append(line_json)

    msg_printer.good("Finished converting sect label file to JSON")
    return output_json


def merge_dictionaries_with_sum(a: Dict, b: Dict) -> Dict:
    # refer to https://stackoverflow.com/questions/11011756/is-there-any-pythonic-way-to-combine-two-dicts-adding-values-for-keys-that-appe?rq=1
    return dict(
        list(a.items()) + list(b.items()) + [(k, a[k] + b[k]) for k in set(b) & set(a)]
    )


def pack_to_length(
    tokenized_text: List[str],
    max_length: int,
    pad_token: str = "<PAD>",
    add_start_end_token: bool = False,
    start_token: str = "<SOS>",
    end_token: str = "<EOS>",
) -> List[str]:
    """ Packs tokenized text to maximum length

    Parameters
    ----------
    tokenized_text : List[str]
        A list of toekns
    max_length : int
        The max length to pack to
    pad_token : int
        The pad token to be used for the padding
    add_start_end_token : bool
        Whether to add the start and end token to every sentence while packing
    start_token : str
        The start token to be used if ``add_start_token`` is True.
    end_token : str
        The end token to be used if ``add_end_token`` is True

    Returns
    -------

    """
    if not add_start_end_token:
        tokenized_text = tokenized_text[:max_length]
    else:
        max_length = max_length if max_length > 2 else 2
        tokenized_text = tokenized_text[: max_length - 2]
        tokenized_text.append(end_token)
        tokenized_text.insert(0, start_token)

    pad_length = max_length - len(tokenized_text)
    for i in range(pad_length):
        tokenized_text.append(pad_token)

    assert len(tokenized_text) == max_length

    return tokenized_text


def download_file(url: str, dest_filename: str) -> None:
    """ Download a file from the given url

    Parameters
    ----------
    url : str
        The url from which the file will be downloaded
    dest_filename : str
        The destination filename

    """
    # NOTE the stream=True parameter below
    msg_printer = Printer()
    block_size = 65536
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    written = 0
    with open(dest_filename, "wb") as f:
        for chunk in tqdm(
            r.iter_content(chunk_size=block_size),
            total=math.ceil(total_size // block_size),
            desc=f"Downloading from {url}",
        ):
            if chunk:  # filter out keep-alive new chunks
                written = written + len(chunk)
                f.write(chunk)
    msg_printer.good(f"Finished downloading {url} to {dest_filename}")


def extract_zip(filename: str, destination_dir: str):
    """ Extracts a zipped file

    Parameters
    ----------
    filename : str
        The zipped filename
    destination_dir : str
        The directory where the zipped will be placed

    """
    msg_printer = Printer()
    try:
        with msg_printer.loading(f"Unzipping file {filename} to {destination_dir}"):
            stdout.flush()
            with zipfile.ZipFile(filename, "r") as z:
                z.extractall(destination_dir)

        msg_printer.good(f"Finished extraction {filename} to {destination_dir}")
    except zipfile.BadZipFile:
        msg_printer.fail("Couldnot extract {filename} to {destination}")


def extract_tar(filename: str, destination_dir: str, mode="r"):
    """ Extracts tar, targz and other files

    Parameters
    ----------
    filename : str
        The tar zipped file
    destination_dir : str
        The destination directory in which the files should be placed
    mode : str
        A valid tar mode. You can refer to https://docs.python.org/3/library/tarfile.html
        for the different modes.

    Returns
    -------

    """
    msg_printer = Printer()
    try:
        with msg_printer.loading(f"Unzipping file {filename} to {destination_dir}"):
            stdout.flush()
            with tarfile.open(filename, mode) as t:
                t.extractall(destination_dir)

        msg_printer.good(f"Finished extraction {filename} to {destination_dir}")
    except tarfile.ExtractError:
        msg_printer.fail("Couldnot extract {filename} to {destination}")


def convert_generic_sect_to_json(filename: str) -> Dict[str, Any]:
    """ Converts the Generic sect data file into more readable json format

        Parameters
        ----------
        filename : str
            The sectlabel file name available at WING-NUS website

        Returns
        -------
        Dict[str, Any]
            text
                The text of the line
            label
                The label of the file
            file_no
                A unique file number
            line_count
                A line count within the file

    """
    file_no = 1
    line_no = 1
    json_dict = {"generic_sect": []}
    with open(filename) as fp:
        for line in fp:
            if bool(line.strip()):
                match_obj = re.search("currHeader=(.*)", line.strip())
                header_label = match_obj.groups()[0]
                header, label = header_label.split(" ")
                header = " ".join(header.split("-"))
                line_no += 1

                json_dict["generic_sect"].append(
                    {
                        "header": header,
                        "label": label,
                        "file_no": file_no,
                        "line_no": line_no,
                    }
                )
            else:
                file_no += 1

    return json_dict


def convert_parscit_to_conll(
    parscit_train_filepath: pathlib.Path
) -> List[Dict[str, Any]]:
    """ Convert the parscit data available at
    "https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/parsCit.train.data"
    to a CONLL dummy version
    This is done so that we can use it with AllenNLPs built in data reader called
    conll2013 dataset reader

    Parameters
    ----------------
    parscit_train_filepath: pathlib.Path
        The path where the train file path is stored

    """
    printer = Printer()
    citation_string = []
    word_tags = []
    output_list = []
    with printer.loading(f"Converting {parscit_train_filepath.name} to conll format"):
        with open(str(parscit_train_filepath), "r", encoding="latin-1") as fp:
            for line in fp:
                if bool(line.strip()):
                    fields = line.strip().split()
                    word = fields[0]
                    tag = fields[-1]
                    word = word.strip()
                    tag = f"{tag.strip()}"
                    word_tag = " ".join([word] + [tag] * 3)
                    citation_string.append(word)
                    word_tags.append(word_tag)
                else:
                    citation_string = " ".join(citation_string)
                    output_list.append(
                        {"word_tags": word_tags, "citation_string": citation_string}
                    )
                    citation_string = []
                    word_tags = []

    printer.good(
        f"Successfully converted {parscit_train_filepath.name} to conll format"
    )
    return output_list


def write_nfold_parscit_train_test(
    parscit_train_filepath: pathlib.Path,
    output_train_filepath: pathlib.Path,
    output_test_filepath: pathlib.Path,
    nsplits: int = 2,
) -> bool:
    """ Convert the parscit train folder into different folds. This is useful for
    n-fold cross validation on the dataset. This method can be iterated over to get
    all the different folds of the data contained in the ``parscit_train_filepath``

    Parameters
    ----------
    parscit_train_filepath : pathlib.Path
        The path where the Parscit file is stored
        The file is available at https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/cora.train
    output_train_filepath : pathlib.Path
        The path where the train fold of the dataset will be stored
    output_test_filepath : pathlib.Path
        The path where the teset fold of the dataset will be stored
    nsplits : int
        The number of splits in the dataset.

    Returns
    -------
    bool
        Indicates whether the particular fold has been written


    """
    citations = convert_parscit_to_conll(parscit_train_filepath=parscit_train_filepath)
    len_citations = len(citations)
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=1729)
    splits = kf.split(np.arange(len_citations))

    train_conll_citations_path = output_train_filepath
    test_conll_citations_path = output_test_filepath

    for train_indices, test_indices in splits:
        train_citations = [citations[train_idx] for train_idx in train_indices]
        test_citations = [citations[test_idx] for test_idx in test_indices]

        # write the file
        with open(train_conll_citations_path, "w") as fp:
            for train_citation in train_citations:
                word_tags = train_citation["word_tags"]
                fp.write("\n".join(word_tags))
                fp.write("\n \n")

        with open(test_conll_citations_path, "w") as fp:
            for test_citation in test_citations:
                word_tags = test_citation["word_tags"]
                fp.write("\n".join(word_tags))
                fp.write("\n \n")

        yield True


def write_cora_to_conll_file(cora_conll_filepath: pathlib.Path) -> None:
    """ Writes cora file that is availabel at https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/cora.train
    to CONLL format

    Parameters
    ----------
    cora_conll_filepath : The destination filepath where the CORA is converted to CONLL format

    """
    citations = convert_parscit_to_conll(pathlib.Path(CORA_FILE))
    with open(cora_conll_filepath, "w") as fp:
        for citation in citations:
            word_tags = citation["word_tags"]
            fp.write("\n".join(word_tags))
            fp.write("\n \n")


def write_parscit_to_conll_file(parscit_conll_filepath: pathlib.Path) -> None:
    """ Write Parscit file to CONLL file format

    Parameters
    ----------
    parscit_conll_filepath : pathlib.Path
        The destination file where the parscit data is written to


    """
    citations = convert_parscit_to_conll(pathlib.Path(PARSCIT_TRAIN_FILE))
    with open(parscit_conll_filepath, "w") as fp:
        for citation in citations:
            word_tags = citation["word_tags"]
            fp.write("\n".join(word_tags))
            fp.write("\n \n")


def pairwise(iterable: Iterable) -> Iterator:
    """ Return the overlapping pairwise elements of the iterable

    Parameters
    ----------
    iterable : Iterable
        Anything that can be iterated

    Returns
    -------
    Iterator
        Iterator over the paired sequence

    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def chunks(seq, n):
    # https://stackoverflow.com/a/312464/190597 (Ned Batchelder)
    """ Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def create_class(classname: str, module_name: str) -> type:
    """ Given the classname and module, creates a class object and returns it

    Parameters
    ----------
    classname : str
        Class name to import
    module_name : str
        The module in which the class is present

    Returns
    -------
    type

    """
    try:
        module = importlib.import_module(module_name)
        try:
            cls = getattr(module, classname)
            return cls
        except AttributeError:
            raise AttributeError(
                f"class {classname} is not found in module {module_name}"
            )
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module {module_name} is not found")


def get_system_mem_in_gb():
    """ Returns the total system memory in GB

    Returns
    -------
    float
        Memory size in GB

    """
    memory_size = psutil.virtual_memory().total
    memory_size = memory_size * 1e-9
    return memory_size
