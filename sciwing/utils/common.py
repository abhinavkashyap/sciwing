from typing import Dict, List, Any, Iterable, Iterator, Union

import math
import requests
from wasabi import Printer
import zipfile
from sys import stdout
import re
from sklearn.model_selection import KFold
import numpy as np
import sciwing.constants as constants
from itertools import tee
import importlib
from tqdm import tqdm
import tarfile
import psutil
import pathlib
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import collections

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

            # if the line is empty then the next line is the beginning of the new file
            if not line:
                file_count += 1
                continue

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


def convert_sectlabel_to_sciwing_clf_format(filename: str, out_dir: str):
    """ Writes the file in the format required for sciwing text classification dataset
    
    Parameters
    ----------
    filename : str
        The path of the sectlabel original format file.
    out_dir : str
        The path where the new files will be written

    Returns
    -------

    """
    texts = []
    labels = []
    with open(filename) as fp:
        for line in tqdm(
            fp, desc="Converting original sect label to Sciwing Classification format"
        ):
            line = line.replace("\n", "")

            if not line:
                continue

            fields = line.split()
            line_content = fields[0]
            line_content = line_content.replace("|||", " ").strip()
            label = fields[-1]
            texts.append(line_content)
            labels.append(label)

    out_dir = pathlib.Path(out_dir)
    train_filename = out_dir.joinpath("sectLabel.train")
    dev_filename = out_dir.joinpath("sectLabel.dev")
    test_filename = out_dir.joinpath("sectLabel.test")

    # TODO: This wot be good for testing sectlabel.
    #  You have to split the lines according to the files they come from.
    #  If the lines are randomly split, then you will lose all the information such as the context of a line

    (
        (train_lines, train_labels),
        (dev_lines, dev_labels),
        (test_lines, test_labels),
    ) = get_train_dev_test_stratified_split(lines=texts, labels=labels)

    with open(train_filename, "w") as fp:
        for text, label in zip(train_lines, train_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")

    with open(dev_filename, "w") as fp:
        for text, label in zip(dev_lines, dev_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")

    with open(test_filename, "w") as fp:
        for text, label in zip(test_lines, test_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")


def convert_generic_sect_to_sciwing_clf_format(filename: str, out_dir: str):
    """ Converts the generic sect original file to the sciwing classification format

    Parameters
    ----------
    filename : str
        The path of the file where the original generic section classification file is stored
    out_dir : str
        The output path where the train, dev and test files are written

    Returns
    -------
    None

    """
    lines = []
    labels = []
    with open(filename) as fp:
        for line in fp:
            if bool(line.strip()):
                match_obj = re.search("currHeader=(.*)", line.strip())
                header_label = match_obj.groups()[0]
                header, label = header_label.split(" ")
                header = " ".join(header.split("-"))
                lines.append(header)
                labels.append(label)

    out_dir = pathlib.Path(out_dir)
    train_filename = out_dir.joinpath("genericSect.train")
    dev_filename = out_dir.joinpath("genericSect.dev")
    test_filename = out_dir.joinpath("genericSect.test")

    (
        (train_lines, train_labels),
        (dev_lines, dev_labels),
        (test_lines, test_labels),
    ) = get_train_dev_test_stratified_split(lines=lines, labels=labels)

    with open(train_filename, "w") as fp:
        for text, label in zip(train_lines, train_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")

    with open(dev_filename, "w") as fp:
        for text, label in zip(dev_lines, dev_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")

    with open(test_filename, "w") as fp:
        for text, label in zip(test_lines, test_labels):
            line = text + "###" + label
            fp.write(line)
            fp.write("\n")


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
        msg_printer.fail(f"Couldnot extract {filename} to {destination_dir}")


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


def convert_parscit_to_conll(
    parscit_train_filepath: pathlib.Path,
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


def convert_parscit_to_sciwing_seqlabel_format(
    parscit_train_filepath: pathlib.Path, output_dir: str
):
    """ Convert the parscit data availabel at
    "https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/parsCit.train.data"
    to the format required for sciwing seqential labelling

    Parameters
    ----------
    parscit_train_filepath : pathlib.Path
        The local path where the files are stored

    output_dir: str
        The output dir where the train dev and test file will be written

    Returns
    -------

    """
    conll_lines = convert_parscit_to_conll(pathlib.Path(parscit_train_filepath))
    instances = []
    for line in conll_lines:
        word_tags = line["word_tags"]
        line_ = []
        for word_tag in word_tags:
            word_tag_ = word_tag.split(" ")
            word = word_tag_[0]
            tag = word_tag_[-1]
            word_tag_ = "###".join([word, tag])
            line_.append(word_tag_)
        instances.append(" ".join(line_))

    # shuffle and split train dev and test
    splitter = ShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1)
    len_citations = len(instances)
    splits = splitter.split(range(len_citations))
    splits = list(splits)
    train_indices, test_indices = splits[0]

    train_instances = [instances[train_idx] for train_idx in train_indices]
    test_instances = [instances[test_idx] for test_idx in test_indices]

    output_dir = pathlib.Path(output_dir)
    train_filepath = output_dir.joinpath("parscit.train")
    dev_filepath = output_dir.joinpath("parscit.dev")
    test_filepath = output_dir.joinpath("parscit.test")

    with open(train_filepath, "w") as fp:
        fp.write("\n".join(train_instances))

    with open(dev_filepath, "w") as fp:
        fp.write("\n".join(test_instances))

    with open(test_filepath, "w") as fp:
        fp.write("\n".join(test_instances))


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


def get_train_dev_test_stratified_split(
    lines: List[str],
    labels: List[str],
    train_split: float = 0.8,
    dev_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 1729,
) -> ((List[str], List[str]), (List[str], List[str]), (List[str], List[str])):
    """ Slits the lines and labels into train, dev and test splits using stratified and
    random shuffle

    Parameters
    ----------
    lines: List[str]
        A list of lines
    labels: List[str]
        A list of labels
    train_split : float
        The proportion of lines to be used for training
    dev_split : float
        The proportion of lines to be used for validation
    test_split : float
        The proportion of lines to be used for testing
    random_state : int
        The seed to be used for randomization. Good for reproducing the same splits
        Passing None will cause the random number generator to be RandomState used by np.random

    Returns
    -------

    """
    len_lines = len(lines)
    len_labels = len(labels)

    assert len_lines == len_labels
    train_test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=dev_split + test_split,
        train_size=train_split,
        random_state=random_state,
    )

    splits = list(train_test_splitter.split(lines, labels))
    train_indices, test_valid_indices = splits[0]

    train_lines = [lines[idx] for idx in train_indices]
    train_labels = [labels[idx] for idx in train_indices]

    test_valid_lines = [lines[idx] for idx in test_valid_indices]
    test_valid_labels = [labels[idx] for idx in test_valid_indices]

    validation_size = dev_split / (test_split + dev_split)
    validation_test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        train_size=1 - validation_size,
        random_state=random_state,
    )

    len_test_valid_lines = len(test_valid_lines)
    len_test_valid_labels = len(test_valid_labels)

    assert len_test_valid_labels == len_test_valid_lines

    test_valid_splits = list(
        validation_test_splitter.split(test_valid_lines, test_valid_labels)
    )

    test_indices, validation_indices = test_valid_splits[0]

    test_lines = [test_valid_lines[idx] for idx in test_indices]
    test_labels = [test_valid_labels[idx] for idx in test_indices]

    validation_lines = [test_valid_lines[idx] for idx in validation_indices]
    validation_labels = [test_valid_labels[idx] for idx in validation_indices]

    return (
        (train_lines, train_labels),
        (validation_lines, validation_labels),
        (test_lines, test_labels),
    )


def cached_path(path: Union[pathlib.Path, str], url: str, unzip=True) -> pathlib.Path:

    if isinstance(path, str):
        path = pathlib.Path(path)
    msg_printer = Printer()
    if path.is_file() or path.is_dir():
        msg_printer.info(f"{path} exists.")
        return path

    download_file(url=url, dest_filename=str(path))

    if unzip:
        if zipfile.is_zipfile(str(path)):
            extract_zip(filename=str(path), destination_dir=str(path.parent))
        if tarfile.is_tarfile(str(path)):
            if "tar" in path.suffix:
                mode = "r"
            elif "gz" in path.suffix:
                mode = "r:gz"
            else:
                mode = "r"

            extract_tar(filename=str(path), destination_dir=str(path.parent), mode=mode)

    return path


def flatten(list_items: List[Any]) -> List[Any]:
    """ Flattens an arbitrarily long nesting of lists

    Parameters
    ----------
    list_items: List[Any]
        It can be an arbitrarily long nesting of lists

    Returns
    -------
    List
        Flattened list
    """
    for el in list_items:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


if __name__ == "__main__":
    data_dir = pathlib.Path(DATA_DIR)
    parscit_train_file = data_dir.joinpath("parsCit.train.data")
    convert_parscit_to_sciwing_seqlabel_format(
        parscit_train_filepath=parscit_train_file, output_dir=str(data_dir)
    )
