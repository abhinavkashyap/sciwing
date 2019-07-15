from parsect.models.parscit_tagger import ParscitTagger
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
from parsect.datasets.parscit_dataset import ParscitDataset
from parsect.metrics.token_cls_accuracy import TokenClassificationAccuracy
from parsect.utils.common import write_parscit_to_conll_file
from parsect.utils.common import write_cora_to_conll_file
import parsect.constants as constants
import os
import torch
import torch.optim as optim
from parsect.engine.engine import Engine
import json
import argparse
import pathlib
import torch.nn as nn
import wasabi
import numpy as np

FILES = constants.FILES
PATHS = constants.PATHS

PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]
CORA_FILE = FILES["CORA_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]
DATA_DIR = PATHS["DATA_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="LSTM CRF Parscit tagger for reference string parsing"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument(
        "--max_num_words",
        help="Maximum number of words to be considered " "in the vocab",
        type=int,
    )
    parser.add_argument(
        "--max_len", help="Maximum length of sentences to be considered", type=int
    )
    parser.add_argument(
        "--max_char_len", help="Maximum length of sentences to be considered", type=int
    )

    parser.add_argument(
        "--debug",
        help="Specify whether this is run on a debug options. The "
        "dataset considered will be small",
        action="store_true",
    )
    parser.add_argument(
        "--debug_dataset_proportion",
        help="The proportion of the dataset " "that will be used if debug is true",
        type=float,
    )
    parser.add_argument("--bs", help="batch size", type=int)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--epochs", help="number of epochs", type=int)
    parser.add_argument(
        "--save_every", help="Save the model every few epochs", type=int
    )
    parser.add_argument(
        "--log_train_metrics_every",
        help="Log training metrics every few iterations",
        type=int,
    )
    parser.add_argument("--emb_dim", help="embedding dimension", type=int)
    parser.add_argument(
        "--char_emb_dim", help="character embedding dimension", type=int
    )
    parser.add_argument(
        "--emb_type",
        help="The type of glove embedding you want. The allowed types are glove_6B_50, glove_6B_100, "
        "glove_6B_200, glove_6B_300",
    )
    parser.add_argument(
        "--hidden_dim", help="Hidden dimension of the lstm encoder", type=int
    )
    parser.add_argument(
        "--bidirectional",
        help="Specify Whether the lstm is bidirectional or uni-directional",
        action="store_true",
    )
    parser.add_argument(
        "--use_char_encoder",
        help="Specify whether to use character encoder with neural-parscit",
        action="store_true",
    )
    parser.add_argument(
        "--char_encoder_hidden_dim",
        help="Character encoder hidden dimension.",
        type=int,
    )
    parser.add_argument(
        "--combine_strategy",
        help="How do you want to combine the hidden dimensions of the two "
        "combinations",
    )

    parser.add_argument("--device", help="Device on which the model is run", type=str)
    args = parser.parse_args()
    msg_printer = wasabi.Printer()

    config = {
        "EXP_NAME": args.exp_name,
        "DEBUG": args.debug,
        "DEBUG_DATASET_PROPORTION": args.debug_dataset_proportion,
        "BATCH_SIZE": args.bs,
        "EMBEDDING_DIMENSION": args.emb_dim,
        "CHAR_EMBEDDING_DIMENSION": args.char_emb_dim,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SAVE_EVERY": args.save_every,
        "LOG_TRAIN_METRICS_EVERY": args.log_train_metrics_every,
        "EMBEDDING_TYPE": args.emb_type,
        "MAX_NUM_WORDS": args.max_num_words,
        "MAX_LENGTH": args.max_len,
        "MAX_CHAR_LENGTH": args.max_char_len,
        "DEVICE": args.device,
        "HIDDEN_DIM": args.hidden_dim,
        "BIDIRECTIONAL": args.bidirectional,
        "COMBINE_STRATEGY": args.combine_strategy,
        "USE_CHAR_ENCODER": args.use_char_encoder,
        "CHAR_ENCODER_HIDDEN_DIM": args.char_encoder_hidden_dim,
    }

    EXP_NAME = config["EXP_NAME"]
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")
    CORA_CONLL_FILE = pathlib.Path(DATA_DIR, "cora_conll.txt")
    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
    CHAR_VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "char_vocab.json")
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    SAVE_EVERY = config["SAVE_EVERY"]
    LOG_TRAIN_METRICS_EVERY = config["LOG_TRAIN_METRICS_EVERY"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    CHAR_EMBEDDING_DIMENSION = config["CHAR_EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config["EMBEDDING_TYPE"]
    TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    DEVICE = config["DEVICE"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    MAX_CHAR_LENGTH = config["MAX_CHAR_LENGTH"]
    USE_CHAR_ENCODER = config["USE_CHAR_ENCODER"]
    CHAR_ENCODER_HIDDEN_DIM = config["CHAR_ENCODER_HIDDEN_DIM"]
    train_conll_filepath = pathlib.Path(DATA_DIR, "parscit_train_conll.txt")
    test_conll_filepath = pathlib.Path(CORA_CONLL_FILE)

    write_parscit_to_conll_file(train_conll_filepath)
    write_cora_to_conll_file(CORA_CONLL_FILE)

    train_dataset = ParscitDataset(
        parscit_conll_file=str(train_conll_filepath),
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        character_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        start_token="<SOS>",
        end_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    validation_dataset = ParscitDataset(
        parscit_conll_file=str(test_conll_filepath),
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        character_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        start_token="<SOS>",
        end_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    test_dataset = ParscitDataset(
        parscit_conll_file=str(CORA_CONLL_FILE),
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        character_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        start_token="<SOS>",
        end_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    train_dataset.get_stats()
    validation_dataset.get_stats()
    test_dataset.get_stats()

    VOCAB_SIZE = train_dataset.word_vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()
    embedding = train_dataset.get_preloaded_word_embedding()
    embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
    char_embedding = train_dataset.get_preloaded_char_embedding()
    char_embedding = nn.Embedding.from_pretrained(char_embedding, freeze=False)
    char_encoder = None

    if USE_CHAR_ENCODER:
        char_encoder = LSTM2VecEncoder(
            emb_dim=CHAR_EMBEDDING_DIMENSION,
            embedding=char_embedding,
            bidirectional=True,
            hidden_dim=CHAR_ENCODER_HIDDEN_DIM,
            combine_strategy="concat",
        )
        EMBEDDING_DIMENSION += 2 * CHAR_ENCODER_HIDDEN_DIM

    lstm2seqencoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embedding,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=True,
        device=torch.device(DEVICE),
    )
    model = ParscitTagger(
        rnn2seqencoder=lstm2seqencoder,
        num_classes=NUM_CLASSES,
        hid_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        character_encoder=char_encoder,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    classnames2idx = train_dataset.classnames2idx
    ignore_indices = [
        classnames2idx["starting"],
        classnames2idx["ending"],
        classnames2idx["padding"],
    ]
    metric = TokenClassificationAccuracy(
        idx2labelname_mapping=train_dataset.idx2classname,
        mask_label_indices=ignore_indices,
    )

    engine = Engine(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        save_dir=MODEL_SAVE_DIR,
        num_epochs=NUM_EPOCHS,
        save_every=SAVE_EVERY,
        log_train_metrics_every=LOG_TRAIN_METRICS_EVERY,
        tensorboard_logdir=TENSORBOARD_LOGDIR,
        device=torch.device(DEVICE),
        metric=metric,
    )

    engine.run()

    config["VOCAB_STORE_LOCATION"] = VOCAB_STORE_LOCATION
    config["CHAR_VOCAB_STORE_LOCATION"] = CHAR_VOCAB_STORE_LOCATION
    config["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR
    config["VOCAB_SIZE"] = VOCAB_SIZE
    config["NUM_CLASSES"] = NUM_CLASSES

    with open(os.path.join(f"{EXP_DIR_PATH}", "config.json"), "w") as fp:
        json.dump(config, fp)
