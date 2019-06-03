from typing import Dict, List
from tqdm import tqdm
import requests
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from wasabi import Printer
import zipfile
from sys import stdout


def convert_secthead_to_json(filename: str) -> Dict:
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
    parsect_json = convert_secthead_to_json(filename)
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

    vocab = Vocab(instances, max_num_words=MAX_NUM_WORDS)

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
    with msg_printer.loading(f'Downloading file {url} to {dest_filename}'):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
    msg_printer.good(f'Finished downloading {url} to {dest_filename}')


def extract_zip(filename: str,
                destination_dir: str):
    msg_printer = Printer()
    try:
        with msg_printer.loading(f'Unzipping file {filename} to {destination_dir}'):
            stdout.flush()
            with zipfile.ZipFile(filename, 'r') as z:
                z.extractall(destination_dir)

        msg_printer.good(f'Finished extraction {filename} to {destination_dir}')
    except zipfile.BadZipFile:
        msg_printer.fail('Couldnot extract {filename} to {destination}')


if __name__ == '__main__':
    import pathlib
    glove_file = pathlib.Path('~/Desktop/glove.6B.zip').expanduser()
    desktop_path = pathlib.Path('~/Desktop').expanduser()
    extract_zip(glove_file, desktop_path)