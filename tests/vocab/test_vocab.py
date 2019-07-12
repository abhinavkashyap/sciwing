import pytest
from parsect.vocab.vocab import Vocab
import os


@pytest.fixture
def instances():
    single_instance = [["i", "like", "nlp", "i", "i", "like"]]
    return {"single_instance": single_instance}


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
            single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == 3 + len(vocab_builder.special_vocab)

    def test_vocab_length_min_freq_1_max_words_0(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 0
        MIN_FREQ = 1

        vocab_builder = Vocab(
            single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == len(vocab_builder.special_vocab)

    def test_vocab_length_min_freq_1_max_words_1(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 1
        MIN_FREQ = 1

        vocab_builder = Vocab(
            single_instance, min_count=MIN_FREQ, max_num_tokens=MAX_NUM_WORDS
        )
        vocab_builder.build_vocab()
        len_vocab = vocab_builder.get_vocab_len()
        assert len_vocab == 1 + len(vocab_builder.special_vocab)

    def test_save_vocab(self, instances, tmpdir):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(single_instance, max_num_tokens=MAX_NUM_WORDS)

        vocab_builder.build_vocab()
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab_builder.save_to_file(vocab_file)

        assert os.path.isfile(vocab_file)

    def test_load_vocab(self, instances, tmpdir):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab_file = tmpdir.mkdir("tempdir").join("vocab.json")
        vocab_builder.save_to_file(vocab_file)

        options, vocab = vocab_builder.load_from_file(vocab_file)

        assert vocab_builder.get_vocab_len() == 3 + len(vocab_builder.special_vocab)

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
            single_instance,
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
        vocab_builder = Vocab(single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        words_freqs = vocab_builder.get_topn_frequent_words(n=1)

        assert words_freqs[0][0] == "i"
        assert words_freqs[0][1] == 3

    def test_print_stats_works(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab_builder = Vocab(single_instance, max_num_tokens=MAX_NUM_WORDS)
        vocab_builder.build_vocab()
        vocab_builder.print_stats()

    def test_load_embedding_has_all_words(self, instances):
        single_instance = instances["single_instance"]
        MAX_NUM_WORDS = 100
        vocab = Vocab(
            instances=single_instance,
            max_num_tokens=MAX_NUM_WORDS,
            embedding_type="glove_6B_50",
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
