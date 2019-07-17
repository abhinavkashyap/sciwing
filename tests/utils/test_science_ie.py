import parsect.constants as constants
import pytest
from parsect.utils.science_ie import ScienceIEDataUtils
import pathlib
from collections import Counter

FILES = constants.FILES
SCIENCE_IE_TRAIN_FOLDER = FILES["SCIENCE_IE_TRAIN_FOLDER"]


@pytest.fixture
def setup_science_ie_train_data_utils():
    utils = ScienceIEDataUtils(pathlib.Path(SCIENCE_IE_TRAIN_FOLDER))
    return utils


class TestScienceIEDataUtils:
    def test_file_ids_different(self, setup_science_ie_train_data_utils):
        utils = setup_science_ie_train_data_utils
        file_ids = utils._get_file_ids()
        counter_file_ids = Counter(file_ids)
        assert all([count == 1 for count in counter_file_ids.values()])

    def test_all_text_read(self, setup_science_ie_train_data_utils):
        utils = setup_science_ie_train_data_utils
        file_ids = utils._get_file_ids()
        for file_id in file_ids:
            text = utils._get_text(file_id)
            assert len(text) > 0

    @pytest.mark.parametrize("entity_type", ["Task", "Process", "Material"])
    def test_get_annotations_for_entity(
        self, setup_science_ie_train_data_utils, entity_type
    ):
        utils = setup_science_ie_train_data_utils
        file_ids = utils._get_file_ids()

        for file_id in file_ids:
            annotations = utils._get_annotations_for_entity(
                file_id=file_id, entity=entity_type
            )
            assert all([len(annotation["words"]) > 0 for annotation in annotations])

    @pytest.mark.parametrize("entity_type", ["Task", "Process", "Material"])
    def test_annotation_id_starts_with_t(
        self, setup_science_ie_train_data_utils, entity_type
    ):
        utils = setup_science_ie_train_data_utils
        file_ids = utils._get_file_ids()

        for file_id in file_ids:
            annotations = utils._get_annotations_for_entity(
                file_id=file_id, entity=entity_type
            )
            assert all(
                [
                    annotation["entity_number"].startswith("T")
                    for annotation in annotations
                ]
            )

    @pytest.mark.parametrize(
        "words, tag, expected_lines, mark_as_O",
        [
            (["word"], "Process", ["word U-Process U-Process U-Process"], False),
            (["word"], "Process", ["word O-Process O-Process O-Process"], True),
            (
                ["word", "word", "word"],
                "Process",
                [
                    "word O-Process O-Process O-Process",
                    "word O-Process O-Process O-Process",
                    "word O-Process O-Process O-Process",
                ],
                True,
            ),
            (["word"], "Process", ["word U-Process U-Process U-Process"], False),
            (
                ["word", "word"],
                "Process",
                [
                    "word B-Process B-Process B-Process",
                    "word L-Process L-Process L-Process",
                ],
                False,
            ),
            (
                ["word", "word", "word"],
                "Process",
                [
                    "word B-Process B-Process B-Process",
                    "word I-Process I-Process I-Process",
                    "word L-Process L-Process L-Process",
                ],
                False,
            ),
        ],
    )
    def test_bilou_for_words(
        self, setup_science_ie_train_data_utils, words, tag, expected_lines, mark_as_O
    ):
        utils = setup_science_ie_train_data_utils
        lines = utils._get_bilou_for_words(words=words, tag=tag, mark_as_O=mark_as_O)
        for idx, line in enumerate(lines):
            print(line)
            assert line == expected_lines[idx]

    @pytest.mark.parametrize(
        "text, annotations, expected_lines",
        [
            (
                "word",
                [{"start": 0, "end": 4, "tag": "Process"}],
                ["word U-Process U-Process U-Process"],
            ),
            (
                "word word",
                [{"start": 0, "end": 4, "tag": "Process"}],
                [
                    "word U-Process U-Process U-Process",
                    "word O-Process O-Process O-Process",
                ],
            ),
            (
                "word. word",
                [{"start": 0, "end": 4, "tag": "Process"}],
                [
                    "word. U-Process U-Process U-Process",
                    "word O-Process O-Process O-Process",
                ],
            ),
            (
                "word word",
                [{"start": 0, "end": 9, "tag": "Process"}],
                [
                    "word B-Process B-Process B-Process",
                    "word L-Process L-Process L-Process",
                ],
            ),
            (
                "word. word",
                [{"start": 0, "end": 10, "tag": "Process"}],
                [
                    "word. B-Process B-Process B-Process",
                    "word L-Process L-Process L-Process",
                ],
            ),
            (
                "word. word word",
                [{"start": 0, "end": 10, "tag": "Process"}],
                [
                    "word. B-Process B-Process B-Process",
                    "word L-Process L-Process L-Process",
                    "word O-Process O-Process O-Process",
                ],
            ),
            (
                "(word) word word",
                [{"start": 1, "end": 5, "tag": "Process"}],
                [
                    "(word) U-Process U-Process U-Process",
                    "word O-Process O-Process O-Process",
                    "word O-Process O-Process O-Process",
                ],
            ),
            (
                "(word) word word",
                [{"start": 1, "end": 16, "tag": "Process"}],
                [
                    "(word) B-Process B-Process B-Process",
                    "word I-Process I-Process I-Process",
                    "word L-Process L-Process L-Process",
                ],
            ),
            (
                "(word) word word",
                [{"start": 1, "end": 16, "tag": "Process"}],
                [
                    "(word) B-Process B-Process B-Process",
                    "word I-Process I-Process I-Process",
                    "word L-Process L-Process L-Process",
                ],
            ),
            (
                "(word) word word",
                [],
                [
                    "(word) O-Process O-Process O-Process",
                    "word O-Process O-Process O-Process",
                    "word O-Process O-Process O-Process",
                ],
            ),
        ],
    )
    def test_private_get_bilou_lines_for_entity(
        self, setup_science_ie_train_data_utils, text, annotations, expected_lines
    ):
        utils = setup_science_ie_train_data_utils
        lines = utils._get_bilou_lines_for_entity(
            text=text, annotations=annotations, entity="Process"
        )
        for idx, line in enumerate(lines):
            assert line == expected_lines[idx]

    @pytest.mark.parametrize("entity_type", ["Task", "Process", "Material"])
    def test_get_bilou_lines(self, setup_science_ie_train_data_utils, entity_type):
        utils = setup_science_ie_train_data_utils
        try:
            file_ids = utils.file_ids
            for file_id in file_ids:
                utils.get_bilou_lines_for_entity(file_id=file_id, entity=entity_type)
        except:
            pytest.fail("Failed to run bilou lines")
