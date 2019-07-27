from parsect.engine.engine import Engine
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.classification.parsect_dataset import ParsectDataset
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from torch.nn import Embedding
import torch.optim as optim
import torch
import numpy as np
import os


import pytest
import parsect.constants as constants

FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


@pytest.fixture
def setup_engine_test_with_simple_classifier(tmpdir):
    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 50
    vocab_store_location = tmpdir.mkdir("tempdir").join("vocab.json")
    DEBUG = True
    BATCH_SIZE = 1
    NUM_TOKENS = 3
    EMB_DIM = 300

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        word_embedding_type="random",
        word_embedding_dimension=EMB_DIM,
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        word_embedding_type="random",
        word_embedding_dimension=EMB_DIM,
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=vocab_store_location,
        debug=DEBUG,
        word_embedding_type="random",
        word_embedding_dimension=EMB_DIM,
    )

    VOCAB_SIZE = MAX_NUM_WORDS + len(train_dataset.vocab.special_vocab)
    NUM_CLASSES = train_dataset.get_num_classes()
    NUM_EPOCHS = 1
    embedding = Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    labels = torch.LongTensor([1])
    metric = PrecisionRecallFMeasure(idx2labelname_mapping=train_dataset.idx2classname)
    embedder = VanillaEmbedder(embedding_dim=EMB_DIM, embedding=embedding)
    encoder = BOW_Encoder(
        emb_dim=EMB_DIM, embedder=embedder, dropout_value=0, aggregation_type="sum"
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=False,
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    engine = Engine(
        model,
        train_dataset,
        validation_dataset,
        test_dataset,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        save_dir=tmpdir.mkdir("model_save"),
        num_epochs=NUM_EPOCHS,
        save_every=1,
        log_train_metrics_every=10,
        metric=metric,
    )

    options = {
        "MAX_NUM_WORDS": MAX_NUM_WORDS,
        "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_TOKENS": NUM_TOKENS,
        "EMB_DIM": EMB_DIM,
        "VOCAB_SIZE": VOCAB_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "NUM_EPOCHS": NUM_EPOCHS,
    }

    return engine, tokens, labels, options


class TestEngine:
    def test_train_loader_gets_equal_length_tokens(
        self, setup_engine_test_with_simple_classifier
    ):
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        train_dataset = engine.get_train_dataset()
        train_loader = engine.get_loader(train_dataset)

        len_tokens = []
        for iter_dict in train_loader:
            tokens = iter_dict["tokens"]
            len_tokens.append(tokens.size()[1])

        # check all lengths are same
        assert len(set(len_tokens)) == 1

    def test_validation_loader_gets_equal_length_tokens(
        self, setup_engine_test_with_simple_classifier
    ):
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        validation_dataset = engine.get_validation_dataset()
        validation_loader = engine.get_loader(validation_dataset)

        len_tokens = []

        for iter_dict in validation_loader:
            tokens = iter_dict["tokens"]
            len_tokens.append(tokens.size()[1])

        assert len(set(len_tokens)) == 1

    def test_loader_gets_equal_length_tokens(
        self, setup_engine_test_with_simple_classifier
    ):
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        test_dataset = engine.get_test_dataset()
        test_loader = engine.get_loader(test_dataset)

        len_tokens = []

        for iter_dict in test_loader:
            tokens = iter_dict["tokens"]
            len_tokens.append(tokens.size()[1])

        assert len(set(len_tokens)) == 1

    def test_one_train_epoch(self, setup_engine_test_with_simple_classifier):
        # check whether you can run train_epoch without throwing an error
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        engine.train_epoch(0)

    def test_save_model(self, setup_engine_test_with_simple_classifier):
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        engine.train_epoch_end(0)

        # test for the file model_epoch_1.pt
        assert os.path.isdir(engine.save_dir)
        assert os.path.isfile(os.path.join(engine.save_dir, "model_epoch_1.pt"))

    def test_runs(self, setup_engine_test_with_simple_classifier):
        """
        Just tests runs without any errors
        """
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        engine.run()

    def test_load_model(self, setup_engine_test_with_simple_classifier):
        """
        Test whether engine loads the model without any error.
        """
        engine, tokens, labels, options = setup_engine_test_with_simple_classifier
        engine.train_epoch_end(0)
        engine.load_model_from_file(
            os.path.join(engine.save_dir, "model_epoch_{0}.pt".format(1))
        )
