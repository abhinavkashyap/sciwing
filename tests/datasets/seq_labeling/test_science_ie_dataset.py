import pytest
import parsect.constants as constants
import pathlib
from parsect.datasets.seq_labeling.science_ie_dataset import ScienceIEDataset
from torch.utils.data import DataLoader
import torch

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
FILES = constants.FILES
SCIENCE_IE_TRAIN_FOLDER = FILES["SCIENCE_IE_TRAIN_FOLDER"]
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]


@pytest.fixture(params=["train", "dev"], scope="session")
def setup_science_ie_dataset(request, tmpdir_factory):
    train_tag = request.param
    task_filename = pathlib.Path(DATA_DIR, f"{train_tag}_task_conll.txt")
    process_filename = pathlib.Path(DATA_DIR, f"{train_tag}_process_conll.txt")
    material_filename = pathlib.Path(DATA_DIR, f"{train_tag}_material_conll.txt")
    out_filename = pathlib.Path(DATA_DIR, f"{train_tag}_science_ie_conll.txt")

    vocab_store_location = tmpdir_factory.mktemp("tempdir").join("vocab.json")
    char_vocab_store_location = tmpdir_factory.mktemp("tempdir_char").join(
        "char_vocab.json"
    )
    DEBUG = False
    MAX_NUM_WORDS = 10000
    MAX_LENGTH = 300
    MAX_CHAR_LENGTH = 25
    EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 25

    dataset = ScienceIEDataset(
        science_ie_conll_file=out_filename,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        word_embedding_type="random",
        word_embedding_dimension=EMBEDDING_DIM,
        word_add_start_end_token=False,
        char_vocab_store_location=char_vocab_store_location,
        character_embedding_dimension=CHAR_EMBEDDING_DIM,
    )
    options = {
        "MAX_NUM_WORDS": MAX_NUM_WORDS,
        "MAX_LENGTH": MAX_LENGTH,
        "MAX_CHAR_LENGTH": MAX_CHAR_LENGTH,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }
    yield dataset, options


class TestScienceIE:
    def test_num_classes(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        num_classes = dataset.get_num_classes()
        assert num_classes == 8

    def test_lines_labels_not_empty(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        lines, labels = dataset.get_lines_labels()

        assert all([bool(line.strip()) for line in lines])
        assert all([bool(label.strip())] for label in labels)

    def test_lines_labels_are_equal_length(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        lines, labels = dataset.get_lines_labels()

        len_lines_labels = zip(
            (len(line.split()) for line in lines),
            (len(label.split()) for label in labels),
        )

        assert all([len_line == len_label for len_line, len_label in len_lines_labels])

    def test_get_stats_works(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset

        try:
            dataset.print_stats()
        except:
            pytest.fail("Failed getting stats for ScienceIE dataset")

    def test_tokens_max_length(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        lines, labels = dataset.get_lines_labels()
        num_lines = len(lines)

        for idx in range(num_lines):
            assert len(dataset[idx]["tokens"]) == options["MAX_LENGTH"]

    def test_char_tokens_max_length(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        lines, labels = dataset.get_lines_labels()
        num_lines = len(lines)
        for idx in range(num_lines):
            char_tokens = dataset[idx]["char_tokens"]
            assert char_tokens.size() == (
                options["MAX_LENGTH"],
                options["MAX_CHAR_LENGTH"],
            )

    def test_instance_dict_with_loader(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
        instances_dict = next(iter(loader))
        assert instances_dict["tokens"].size() == (2, options["MAX_LENGTH"])
        assert instances_dict["label"].size() == (2, 3 * options["MAX_LENGTH"])

    def test_labels_obey_rules(self, setup_science_ie_dataset):
        dataset, options = setup_science_ie_dataset
        instances = dataset.word_instances
        len_instances = len(instances)

        for idx in range(len_instances):
            iter_dict = dataset[idx]
            label = iter_dict["label"]
            task_label, process_label, material_label = torch.chunk(
                label, chunks=3, dim=0
            )

            # task label should be in [0, 7]
            assert torch.all(torch.ge(task_label, 0) & torch.le(task_label, 7)).item()

            # process label should be in [8, 15]
            assert torch.all(
                torch.ge(process_label, 8) & torch.le(process_label, 15)
            ).item()

            assert torch.all(
                torch.ge(material_label, 15) & torch.le(material_label, 23)
            ).item()
