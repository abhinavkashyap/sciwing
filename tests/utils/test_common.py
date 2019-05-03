import parsect.constants as constants
from parsect.utils.common import convert_secthead_to_json
FILES = constants.FILES

SECTLABEL_FILENAME = FILES['SECT_LABEL_FILE']


class TestCommon:
    """
    1. Test to make sure the dicts are full
    2. Test to count the number of files - 40 papers
    """

    def test_text_not_empty(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json['parse_sect']

        text = [bool(each_line['text']) for each_line in output]
        assert all(text)

    def test_label_not_empty(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json['parse_sect']

        labels = [bool(each_line['label']) for each_line in output]
        assert all(labels)

    def test_filecount(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json['parse_sect']

        file_numbers = [each_line['file_no'] for each_line in output]
        file_numbers = set(file_numbers)
        assert len(file_numbers) == 40  # number of files expected
