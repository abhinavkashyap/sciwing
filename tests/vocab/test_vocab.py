import pytest
from sciwing.vocab.vocab import Vocab
import os
from sciwing.utils.common import get_system_mem_in_gb


@pytest.fixture
def instances():
    single_instance = [["i", "like", "nlp", "i", "i", "like"]]
    return {"single_instance": single_instance}


system_mem = int(get_system_mem_in_gb())


class TestVocab:
    def test_build_vocab_single_instance_has_words(self, instances):
        single_instance = instances["single_instance"]
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=1000)
        vocab = vocab_builder.map_tokens_to_freq_idx()

        assert "i" in vocab.keys()
        assert "like" in vocab.keys()
        assert "nlp" in vocab.keys()

    def test_build_vocab_single_instance_descending_order(self, instances):
        single_instance = instances["single_instance"]
        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=1000, min_count=1
        )
        vocab = vocab_builder.map_tokens_to_freq_idx()

        i_freq, i_idx = vocab["i"]
        like_freq, like_idx = vocab["like"]
        nlp_freq, nlp_idx = vocab["nlp"]

        # since 'i' appears more number of times than 'like' appears more
        # number of times than nlp
        assert i_idx < like_idx < nlp_idx
        assert i_freq > like_freq > nlp_freq

    def test_vocab_always_has_special_tokens(self, instances):
        single_instance = instances["single_instance"]
        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=1000, min_count=1
        )

        vocab = vocab_builder.map_tokens_to_freq_idx()
        assert vocab_builder.unk_token in vocab.keys()
        assert vocab_builder.pad_token in vocab.keys()
        assert vocab_builder.start_token in vocab.keys()
        assert vocab_builder.end_token in vocab.keys()

    def test_single_instance_min_count(self, instances):
        single_instance = instances["single_instance"]

        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=1000, min_count=2
        )
        vocab_builder.build_vocab()
        vocab = vocab_builder.map_tokens_to_freq_idx()
        vocab = vocab_builder.clip_on_mincount(vocab)

        # check that is mapped to unk
        nlp_freq, nlp_idx = vocab["nlp"]
        assert nlp_idx == vocab_builder.token2idx["<UNK>"]

    def test_single_instance_clip_on_max_num(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 1
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab = vocab_builder.map_tokens_to_freq_idx()

        vocab = vocab_builder.clip_on_max_num(vocab)

        vocab_len = len(set(idx for freq, idx in vocab.values()))

        assert vocab_len == MAX_NUM_WORDS + len(vocab_builder.special_vocab)

    def test_single_instance_build_vocab(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        MIN_FREQ = 1
        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=MAX_NUM_WORDS, min_count=MIN_FREQ
        )

        vocab = vocab_builder.build_vocab()

        assert "i" in vocab.keys()
        assert "like" in vocab.keys()
        assert "nlp" in vocab.keys()

        vocab_len = len(set(idx for freq, idx in vocab.values()))

        assert vocab_len == 3 + len(vocab_builder.special_vocab)

    def test_build_vocab_single_instance_min_freq_2(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        MIN_FREQ = 2
        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=MAX_NUM_WORDS, min_count=MIN_FREQ
        )
        vocab = vocab_builder.build_vocab()

        vocab_len = len(set(idx for freq, idx in vocab.values()))

        assert vocab_len == 2 + len(vocab_builder.special_vocab)

    def test_build_vocab_single_instance_max_size_1(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 1
        MIN_FREQ = 1

        vocab_builder = Vocab(
            instances=single_instance, max_num_tokens=MAX_NUM_WORDS, min_count=MIN_FREQ
        )
        vocab = vocab_builder.build_vocab()

        vocab_len = len(set(idx for freq, idx in vocab.values()))

        assert vocab_len == 1 + len(vocab_builder.special_vocab)

    def test_vocab_length_min_freq_1_max_words_100(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        MIN_FREQ = 1

        vocab_builder = Vocab(
            instances=single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == 3 + len(vocab_builder.special_vocab)

    def test_vocab_length_min_freq_1_max_words_0(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 0
        MIN_FREQ = 1

        vocab_builder = Vocab(
            instances=single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == len(vocab_builder.special_vocab)

    def test_vocab_length_min_freq_1_max_words_1(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 1
        MIN_FREQ = 1

        vocab_builder = Vocab(
            instances=single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == 1 + len(vocab_builder.special_vocab)

    def test_save_vocab(self, instances, tmpdir):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)

        vocab_builder.build_vocab()
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab_builder.save_to_file(vocab_file)

        assert os.path.isfile(vocab_file)

    def test_load_vocab(self, instances, tmpdir):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab_builder.save_to_file(vocab_file)

        vocab = Vocab.load_from_file(filename=vocab_file)

        assert vocab.get_vocab_len() == 3 + len(vocab_builder.special_vocab)

    @pytest.mark.parametrize(
        "start_token,end_token,unk_token,pad_token",
        [("<SOS>", "<EOS>", "<UNK>", "<PAD>"), (" ", " ", " ", " ")],
    )
    def test_idx2token(self, instances, start_token, end_token, unk_token, pad_token):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        vocab_builder.build_vocab()
        idx2token = vocab_builder.idx2token
        len_idx2token = len(idx2token)
        indices = idx2token.keys()
        indices = sorted(indices)

        # tests all indices are in order
        assert indices == list(range(len_idx2token))

    def test_idx2token_for_unk(self, instances):
        """" Many words map to UNK in the vocab. For example say the index for UNK is 3.
        Then mapping 3 to the token should always map to UNK and not any other word
        """
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            start_token="<SOS>",
            end_token="<EOS>",
            pad_token="<PAD>",
            unk_token="<UNK>",
        )
        vocab_builder.build_vocab()
        UNK_IDX = vocab_builder.special_vocab[vocab_builder.unk_token][1]
        assert vocab_builder.get_token_from_idx(UNK_IDX) == "<UNK>"

    def test_idx2token_out_of_bounds(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        print(vocab_builder.get_idx2token_mapping())
        with pytest.raises(ValueError):
            vocab_builder.get_token_from_idx(100)

    def test_idx2token_cries_for_vocab(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        with pytest.raises(ValueError):
            vocab_builder.get_idx_from_token(1)

    @pytest.mark.parametrize(
        "start_token,end_token,unk_token,pad_token",
        [("<SOS>", "<EOS>", "<UNK>", "<PAD>"), (" ", " ", " ", " ")],
    )
    def test_token2idx(self, instances, start_token, end_token, unk_token, pad_token):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        vocab_builder.build_vocab()
        token2idx = vocab_builder.token2idx
        len_indices = len(token2idx.keys())
        indices = token2idx.values()
        indices = sorted(indices)
        assert indices == list(range(len_indices))

    def test_orig_vocab_len(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 0
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab_len = vocab_builder.get_orig_vocab_len()
        assert vocab_len == 3 + len(vocab_builder.special_vocab)

    def test_get_topn(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        words_freqs = vocab_builder.get_topn_frequent_words(n=1)

        assert words_freqs[0][0] == "i"
        assert words_freqs[0][1] == 3

    def test_print_stats_works(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab_builder.print_stats()

    @pytest.mark.skipif(
        system_mem < 16, reason="Cannot loading embeddings because memory is low"
    )
    @pytest.mark.parametrize(
        "embedding_type",
        ["glove_6B_50", "glove_6B_100", "glove_6B_200", "glove_6B_300", "random"],
    )
    def test_load_embedding_has_all_words(self, instances, embedding_type):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type=embedding_type,
        )
        vocab.build_vocab()
        embedding = vocab.load_embedding()
        assert embedding.size(0) == vocab.get_vocab_len()

    def test_random_embeddinng_has_2dimensions(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type=None,
            embedding_dimension=300,
        )
        vocab.build_vocab()
        embeddings = vocab.load_embedding()
        assert embeddings.ndimension() == 2

    @pytest.mark.parametrize("save_vocab", [True, False])
    def test_add_token(self, instances, tmpdir, save_vocab):
        instance_dict = instances
        single_instance = instance_dict["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type=None,
            embedding_dimension=300,
            store_location=vocab_file,
        )
        vocab.build_vocab()
        vocab._add_token("very", save_vocab=save_vocab)

        assert "very" in vocab.vocab.keys()
        assert vocab.vocab["very"] == (1, 7)
        assert vocab.token2idx["very"] == 7
        assert vocab.idx2token[7] == "very"

    def test_add_tokens(self, instances, tmpdir):
        instance_dict = instances
        single_instance = instance_dict["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type=None,
            embedding_dimension=300,
            store_location=vocab_file,
        )
        vocab.build_vocab()
        vocab.add_tokens(["very", "much"])
        assert "very" in vocab.vocab.keys()
        assert "much" in vocab.vocab.keys()
        assert vocab.vocab["very"] == (1, 7)
        assert vocab.vocab["much"] == (1, 8)
        assert vocab.get_token_from_idx(7) == "very"
        assert vocab.get_token_from_idx(8) == "much"
        assert vocab.get_idx_from_token("very") == 7
        assert vocab.get_idx_from_token("much") == 8

    def test_disp_sentences_from_indices(self, instances, tmpdir):
        instance_dict = instances
        single_instance = instance_dict["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type=None,
            embedding_dimension=300,
            store_location=vocab_file,
        )
        vocab.build_vocab()
        sent = vocab.get_disp_sentence_from_indices([0, 1, 2, 3])
        assert type(sent) is str

    def test_max_num_tokens_unset(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = None
        vocab = Vocab(instances=single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab.build_vocab()
        assert vocab.max_num_tokens == 3 + len(vocab.special_vocab.keys())

    def test_max_instance_length(self, instances):
        single_instance = instances["single_instance"]
        vocab_builder = Vocab(instances=single_instance, max_num_tokens=1000)
        assert vocab_builder.max_instance_length == 100
