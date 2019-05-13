from typing import Dict
from tqdm import tqdm
import json
from parsect.tokenizers.word_tokenizer import WordTokenizer
from parsect.vocab.vocab import Vocab
from parsect.numericalizer.numericalizer import Numericalizer
from wasabi import Printer


def convert_secthead_to_json(filename: str) -> Dict:
    """
    Converts the secthead file into json format
    :return:
    """
    file_count = 1
    line_count = 1
    output_json = {"parse_sect": []}
    msg_printer = Printer()

    msg_printer.divider('Converting SectLabel File to JSON')
    with open(filename) as fp:
        for line in tqdm(fp, desc="Converting SectLabel File to JSON"):
            line = line.replace('\n', '')

            # if the line is empty then the next line is the beginning of the
            if not line:
                file_count += 1
                continue

            # new file
            fields = line.split()
            line_content = fields[0]  # first column contains the content text
            line_content = line_content.replace('|||', ' ')  # every word in the line is sepearted by |||
            label = fields[-1]  # the last column contains the field marked
            line_json = {
                'text': line_content,
                'label': label,
                'file_no': file_count,
                'line_count': line_count
            }
            line_count += 1

            output_json['parse_sect'].append(line_json)

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
    parsect_lines = parsect_json['parse_sect']

    print("*" * 80)
    print('TOKENIZATION')
    print("*" * 80)
    tokenizer = WordTokenizer()

    lines = []
    labels = []

    for line_json in tqdm(parsect_lines, desc="READING SECT LABEL LINES",
                          total=len(parsect_lines)):
        text = line_json['text']
        label = line_json['label']
        lines.append(text)
        labels.append(label)

    instances = tokenizer.tokenize_batch(lines)
    num_instances = len(instances)

    MAX_NUM_WORDS = 3000
    MAX_LENGTH = 15

    print("*" * 80)
    print('VOCAB')
    print("*" * 80)

    vocab = Vocab(instances, max_num_words=MAX_NUM_WORDS)

    print("*" * 80)
    print('NUMERICALIZATION')
    print("*" * 80)

    numericalizer = Numericalizer(max_length=MAX_LENGTH,
                                  vocabulary=vocab)

    lengths, numericalized_instances = numericalizer.numericalize_batch_instances(instances)

    output_json = {'parse_sect': []}

    for idx in tqdm(range(num_instances), desc="Forming output json",
                              total=num_instances):
        line_json = parsect_lines[idx]
        text = line_json['text']
        label = line_json['label']
        file_no = line_json['file_no']
        line_count = line_json['line_count']
        length = lengths[idx]
        numericalized_token = numericalized_instances[idx]
        output_json['parse_sect'].append({
            'text': text,
            'label': label,
            'length': length,
            'tokenized_text': numericalized_token,
            'file_no': file_no,
            'line_count': line_count
        })

    return output_json


if __name__ == '__main__':
    import os
    import parsect.constants as constants
    import json
    PATHS = constants.PATHS
    DATA_DIR = PATHS['DATA_DIR']
    filename = "sectLabel.train.data"
    filename = os.path.join(DATA_DIR, filename)
    secthead_json = write_tokenization_vis_json(filename)

    with open(os.path.join(DATA_DIR, 'sectLabel.tokenized.train.json'), 'w') as fp:
        json.dump(secthead_json, fp)
