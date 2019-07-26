import pytest
from parsect.vocab.word_emb_loader import WordEmbLoader
from parsect.vocab.vocab import Vocab
import os


@pytest.fixture()
def setup_word_emb_loader():
    instances = [["load", "vocab"]]
    vocab = Vocab(instances=instances, max_num_tokens=1000)
    vocab.build_vocab()
    return vocab


class TestWordEmbLoader:
    def test_invalid_embedding_type(self, setup_word_emb_loader):
        vocab = setup_word_emb_loader
        with pytest.raises(AssertionError):
            emb_loader = WordEmbLoader(
                token2idx=vocab.get_token2idx_mapping(), embedding_type="notexistent"
            )

    @pytest.mark.parametrize(
        "embedding_type",
        ["glove_6B_50", "glove_6B_100", "glove_6B_200", "glove_6B_300", "parscit"],
    )
    def test_preloaded_file_exists(self, setup_word_emb_loader, embedding_type):
        vocab = setup_word_emb_loader

        emb_loader = WordEmbLoader(
            token2idx=vocab.get_token2idx_mapping(), embedding_type=embedding_type
        )
        preloaded_filename = emb_loader.get_preloaded_filename()

        assert os.path.isfile(preloaded_filename)

    @pytest.mark.parametrize(
        "embedding_type",
        [
            "glove_6B_50",
            "glove_6B_100",
            "glove_6B_200",
            "glove_6B_300",
            "parscit",
            "random",
        ],
    )
    def test_all_vocab_words_have_embedding(
        self, setup_word_emb_loader, embedding_type
    ):
        vocab = setup_word_emb_loader
        emb_loader = WordEmbLoader(
            token2idx=vocab.get_token2idx_mapping(), embedding_type=embedding_type
        )

        vocab_embedding = emb_loader.vocab_embedding

        words = vocab_embedding.keys()

        assert len(words) == vocab.get_vocab_len()
