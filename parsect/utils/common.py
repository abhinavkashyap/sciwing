from typing import Dict
from tqdm import tqdm


def convert_secthead_to_json(filename: str) -> Dict:
    """
    Converts the secthead file into json format
    :return:
    """
    file_count = 1
    line_count = 1
    output_json = {"parse_sect": []}
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

    return output_json


if __name__ == '__main__':
    import os
    import parsect.constants as constants
    import json
    PATHS = constants.PATHS
    DATA_DIR = PATHS['DATA_DIR']
    filename = "sectLabel.train.data"
    filename = os.path.join(DATA_DIR, filename)
    secthead_json = convert_secthead_to_json(filename)

    with open(os.path.join(DATA_DIR, 'sectLabel.train.json'), 'w') as fp:
        json.dump(secthead_json, fp)
