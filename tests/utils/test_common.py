import parsect.constants as constants
from parsect.utils.common import convert_secthead_to_json
from parsect.utils.common import merge_dictionaries_with_sum
from parsect.utils.common import pack_to_length

FILES = constants.FILES

SECTLABEL_FILENAME = FILES["SECT_LABEL_FILE"]


class TestCommon:
    """
    1. Test to make sure the dicts are full
    2. Test to count the number of files - 40 papers
    """

    def test_text_not_empty(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json["parse_sect"]

        text = [bool(each_line["text"]) for each_line in output]
        assert all(text)

    def test_label_not_empty(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json["parse_sect"]

        labels = [bool(each_line["label"]) for each_line in output]
        assert all(labels)

    def test_filecount(self):
        output_json = convert_secthead_to_json(SECTLABEL_FILENAME)
        output = output_json["parse_sect"]

        file_numbers = [each_line["file_no"] for each_line in output]
        file_numbers = set(file_numbers)
        assert len(file_numbers) == 40  # number of files expected

    def test_merge_dictionaries_empty(self):
        a = {}
        b = {"a": 0, "b": 1}
        a_b = merge_dictionaries_with_sum(a, b)
        expected = {"a": 0, "b": 1}
        assert a_b == expected

    def test_merge_dictionaries_one_zero(self):
        a = {"a": 0}
        b = {"a": 0, "b": 1}
        a_b = merge_dictionaries_with_sum(a, b)
        expected = {"a": 0, "b": 1}
        assert a_b == expected

    def test_merge_two_full_dictionaries(self):
        a = {"a": 1, "b": 2}
        b = {"a": 2, "b": 4, "c": 0}
        expected = {"a": 3, "b": 6, "c": 0}
        a_b = merge_dictionaries_with_sum(a, b)
        assert a_b == expected

    def test_pack_to_length_text_lower(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 10
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=False,
        )
        assert (
            tokenized_text_padded
            == ["i", "am", "going", "to", "write", "tests"] + ["<PAD>"] * 4
        )

    def test_pack_to_length_text_higher(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 3
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=False,
        )
        assert tokenized_text_padded == ["i", "am", "going"]

    def test_pack_to_length_text_equal(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 6
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=False,
        )
        assert tokenized_text_padded == tokenized_text

    def test_pack_to_length_text_lower_with_start_end_token(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 10
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=True,
            start_token="<SOS>",
            end_token="<EOS>",
        )
        assert (
            tokenized_text_padded
            == ["<SOS>", "i", "am", "going", "to", "write", "tests", "<EOS>"]
            + ["<PAD>"] * 2
        )

    def test_pack_to_length_text_higher_with_start_end_token(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 3
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=True,
            start_token="<SOS>",
            end_token="<EOS>",
        )
        assert tokenized_text_padded == ["<SOS>", "i", "<EOS>"]

    def test_pack_to_length_text_equal_with_start_end_token(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 6
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=True,
            start_token="<SOS>",
            end_token="<EOS>",
        )
        assert tokenized_text_padded == ["<SOS>", "i", "am", "going", "to", "<EOS>"]

    def test_pack_to_length_max_length_lt_2_add_start_end_token(self):
        tokenized_text = ["i", "am", "going", "to", "write", "tests"]
        length = 2
        tokenized_text_padded = pack_to_length(
            tokenized_text=tokenized_text,
            max_length=length,
            pad_token="<PAD>",
            add_start_end_token=True,
            start_token="<SOS>",
            end_token="<EOS>",
        )

        assert tokenized_text_padded == ["<SOS>", "<EOS>"]
