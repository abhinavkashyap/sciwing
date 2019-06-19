from parsect.models.bert_seq_classifier import BertSeqClassifier
from parsect.datasets.parsect_dataset import ParsectDataset
import parsect.constants as constants
import os
import torch
import torch.optim as optim
from parsect.engine.engine import Engine
import json
import argparse
import wasabi
from pytorch_pretrained_bert.tokenization import BertTokenizer

FILES = constants.FILES
PATHS = constants.PATHS

SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Finetuning the task classifier using Bert"
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
        "--emb_type",
        help="The type of glove embedding you want. The allowed types are glove_6B_50, glove_6B_100, "
        "glove_6B_200, glove_6B_300",
    )
    parser.add_argument(
        "--bert_type",
        help="Specify the bert model to be used. One of bert-base-uncased, bert-base-cased, "
        "bert-large-uncased, bert-large-cased can be used",
    )
    parser.add_argument("--device", help="Device on which the model is run", type=str)
    args = parser.parse_args()

    config = {
        "EXP_NAME": args.exp_name,
        "DEBUG": args.debug,
        "DEBUG_DATASET_PROPORTION": args.debug_dataset_proportion,
        "BATCH_SIZE": args.bs,
        "EMBEDDING_DIMENSION": args.emb_dim,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SAVE_EVERY": args.save_every,
        "LOG_TRAIN_METRICS_EVERY": args.log_train_metrics_every,
        "EMBEDDING_TYPE": args.emb_type,
        "BERT_TYPE": args.bert_type,
        "MAX_NUM_WORDS": args.max_num_words,
        "MAX_LENGTH": args.max_len,
        "DEVICE": args.device,
    }

    EXP_NAME = config["EXP_NAME"]
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")
    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    SAVE_EVERY = config["SAVE_EVERY"]
    LOG_TRAIN_METRICS_EVERY = config["LOG_TRAIN_METRICS_EVERY"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config["EMBEDDING_TYPE"]
    TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)
    BERT_TYPE = config["BERT_TYPE"]
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    DEVICE = config["DEVICE"]

    msg_printer = wasabi.Printer()
    with msg_printer.loading(f"Loading Bert Tokenizer type {BERT_TYPE}"):
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
        tokenization_type="bert",
        tokenizer=bert_tokenizer,
        start_token="[CLS]",
        end_token="[SEP]",
        pad_token="[PAD]",
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
        tokenizer=bert_tokenizer,
        tokenization_type="bert",
        start_token="[CLS]",
        end_token="[SEP]",
        pad_token="[PAD]",
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIMENSION,
        tokenizer=bert_tokenizer,
        tokenization_type="bert",
        start_token="[CLS]",
        end_token="[SEP]",
        pad_token="[PAD]",
    )

    VOCAB_SIZE = train_dataset.vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()

    model = BertSeqClassifier(
        num_classes=NUM_CLASSES,
        emb_dim=EMBEDDING_DIMENSION,
        dropout_value=0.0,
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE),
    )

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
    )

    engine.run()

    config["VOCAB_STORE_LOCATION"] = VOCAB_STORE_LOCATION
    config["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR
    config["VOCAB_SIZE"] = VOCAB_SIZE
    config["NUM_CLASSES"] = NUM_CLASSES
    with open(os.path.join(EXP_DIR_PATH, "config.json"), "w") as fp:
        json.dump(config, fp)
