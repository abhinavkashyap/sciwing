from typing import Dict, List, Any, Iterable, Iterator
from tqdm import tqdm
import requests
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from wasabi import Printer
import zipfile
from sys import stdout
import re
import pathlib
from sklearn.model_selection import KFold
import numpy as np
import parsect.constants as constants
from itertools import tee

PATHS = constants.PATHS
FILES = constants.FILES
DATA_DIR = PATHS["DATA_DIR"]
CORA_FILE = FILES["CORA_FILE"]
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]


def convert_sectlabel_to_json(filename: str) -> Dict:
    """
    Converts the secthead file into json format
    :return:
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


def write_tokenization_vis_json(filename: str) -> Dict:
    """
    takes the parse sect data file and converts and
    numericalization is done. The data is converted to json
    for visualization
    :param filename: str
    json file name where List[Dict[text, label]] are stored
    """
    parsect_json = convert_sectlabel_to_json(filename)
    parsect_lines = parsect_json["parse_sect"]

    print("*" * 80)
    print("TOKENIZATION")
    print("*" * 80)
    tokenizer = WordTokenizer()

    lines = []
    labels = []

    for line_json in tqdm(
        parsect_lines, desc="READING SECT LABEL LINES", total=len(parsect_lines)
    ):
        text = line_json["text"]
        label = line_json["label"]
        lines.append(text)
        labels.append(label)

    instances = tokenizer.tokenize_batch(lines)
    num_instances = len(instances)

    MAX_NUM_WORDS = 3000
    MAX_LENGTH = 15

    print("*" * 80)
    print("VOCAB")
    print("*" * 80)

    vocab = Vocab(instances, max_num_tokens=MAX_NUM_WORDS)

    print("*" * 80)
    print("NUMERICALIZATION")
    print("*" * 80)

    numericalizer = Numericalizer(max_length=MAX_LENGTH, vocabulary=vocab)

    lengths, numericalized_instances = numericalizer.numericalize_batch_instances(
        instances
    )

    output_json = {"parse_sect": []}

    for idx in tqdm(
        range(num_instances), desc="Forming output json", total=num_instances
    ):
        line_json = parsect_lines[idx]
        text = line_json["text"]
        label = line_json["label"]
        file_no = line_json["file_no"]
        line_count = line_json["line_count"]
        length = lengths[idx]
        numericalized_token = numericalized_instances[idx]
        output_json["parse_sect"].append(
            {
                "text": text,
                "label": label,
                "length": length,
                "tokenized_text": numericalized_token,
                "file_no": file_no,
                "line_count": line_count,
            }
        )

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
    # NOTE the stream=True parameter below
    msg_printer = Printer()
    with msg_printer.loading(f"Downloading file {url} to {dest_filename}"):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
    msg_printer.good(f"Finished downloading {url} to {dest_filename}")


def extract_zip(filename: str, destination_dir: str):
    msg_printer = Printer()
    try:
        with msg_printer.loading(f"Unzipping file {filename} to {destination_dir}"):
            stdout.flush()
            with zipfile.ZipFile(filename, "r") as z:
                z.extractall(destination_dir)

        msg_printer.good(f"Finished extraction {filename} to {destination_dir}")
    except zipfile.BadZipFile:
        msg_printer.fail("Couldnot extract {filename} to {destination}")


def convert_generic_sect_to_json(filename: str) -> Dict[str, Any]:
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
    """
    Convert the parscit data available at
    "https://github.com/knmnyn/ParsCit/blob/master/crfpp/traindata/parsCit.train.data"
    to a CONLL dummy version
    This is done so that we can use it with AllenNLPs built in data reader called
    conll2013 dataset reader
    :param parscit_train_filepath: type: pathlib.Path
    The path where the train file path is stored
    :return: None
    """
    printer = Printer()
    citation_string = []
    word_tags = []
    output_list = []
    with printer.loading(f"Converting {parscit_train_filepath.name} to conll format"):
        with open(str(parscit_train_filepath), "r", encoding="utf-8") as fp:
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
    parscit_train_filepath: pathlib.Path, nsplits: int = 2
) -> bool:
    """
    This method writes different train and test citations file that can be given
    to conll2013 dataset reader of allennlp
    This method is mainly used to train the lstm-crf model
    :param parscit_train_filepath: type: pathlib.Path
    :param nsplits: type: int
    Number of kfold splits.
    :return: bool
    Indicates that the file has been written
    """
    citations = convert_parscit_to_conll(parscit_train_filepath=parscit_train_filepath)
    len_citations = len(citations)
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=1729)
    splits = kf.split(np.arange(len_citations))
    data_dir = pathlib.Path(DATA_DIR)

    train_conll_citations_path = data_dir.joinpath("parscit_train_conll.txt")
    test_conll_citations_path = data_dir.joinpath("parscit_test_conll.txt")

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
    citations = convert_parscit_to_conll(pathlib.Path(CORA_FILE))
    with open(cora_conll_filepath, "w") as fp:
        for citation in citations:
            word_tags = citation["word_tags"]
            fp.write("\n".join(word_tags))
            fp.write("\n \n")


def write_parscit_to_conll_file(parscit_conll_filepath: pathlib.Path) -> None:
    citations = convert_parscit_to_conll(pathlib.Path(PARSCIT_TRAIN_FILE))
    with open(parscit_conll_filepath, "w") as fp:
        for citation in citations:
            word_tags = citation["word_tags"]
            fp.write("\n".join(word_tags))
            fp.write("\n \n")


def pairwise(iterable: Iterable) -> Iterator:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
