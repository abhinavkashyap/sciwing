import pytest
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.data.line import Line
from sciwing.tokenizers.word_tokenizer import WordTokenizer
from sciwing.tokenizers.character_tokenizer import CharacterTokenizer
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import torch


@pytest.fixture(scope="session")
def clf_dataset_manager(tmpdir_factory):
    train_file = tmpdir_factory.mktemp("train_data").join("train_file.txt")
    train_file.write("train_line1###label1\ntrain_line2###label2")

    dev_file = tmpdir_factory.mktemp("dev_data").join("dev_file.txt")
    dev_file.write("dev_line1###label1\ndev_line2###label2")

    test_file = tmpdir_factory.mktemp("test_data").join("test_file.txt")
    test_file.write("test_line1###label1\ntest_line2###label2")

    clf_dataset_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
        batch_size=1,
    )

    return clf_dataset_manager


@pytest.fixture(params=[(10, 100)])
def setup_char_embedder(request, clf_dataset_manager):
    char_embedding_dim, hidden_dim = request.param
    datset_manager = clf_dataset_manager
    embedder = CharEmbedder(
        char_embedding_dimension=char_embedding_dim,
        hidden_dimension=hidden_dim,
        datasets_manager=datset_manager,
    )
    texts = ["This is sentence", "This is another sentence"]
    lines = []
    for text in texts:
        line = Line(
            text=text,
            tokenizers={"tokens": WordTokenizer(), "char_tokens": CharacterTokenizer()},
        )
        lines.append(line)

    return embedder, lines


class TestCharEmbedder:
    def test_encoding_dimension(self, setup_char_embedder):
        embedder, lines = setup_char_embedder
        embedded = embedder(lines)
        assert embedded.size(0) == 2
        assert embedded.size(2) == embedder.hidden_dimension * 2

    def test_embedding_set_lines(self, setup_char_embedder):
        embedder, lines = setup_char_embedder
        _ = embedder(lines)
        for line in lines:
            tokens = line.tokens["tokens"]
            for token in tokens:
                assert isinstance(
                    token.get_embedding("char_embedding"), torch.FloatTensor
                )
