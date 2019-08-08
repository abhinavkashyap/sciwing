from parsect.datasets.classification.parsect_dataset import ParsectDataset
from parsect.models.elmo_lstm_classifier import ElmoLSTMClassifier
from parsect.modules.embedders.elmo_embedder import ElmoEmbedder
from parsect.modules.elmo_lstm_encoder import ElmoLSTMEncoder
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import parsect.constants as constants
import os
import torch.nn as nn

import torch.optim as optim
from parsect.engine.engine import Engine
import json
import argparse
import torch


FILES = constants.FILES
PATHS = constants.PATHS

SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Elmo with LSTM encoder followed by linear classifier "
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)

    parser.add_argument(
        "--device", help="Specify the device where the model is run", type=str
    )

    parser.add_argument(
        "--max_num_words",
        help="Maximum number of words to be considered " "in the vocab",
        type=int,
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
        "glove_6B_200, glove_6B_300, random",
    )
    parser.add_argument(
        "--hidden_dim", help="Hidden dimension of the LSTM network", type=int
    )
    parser.add_argument(
        "--max_length", help="Maximum length of the inputs to the encoder", type=int
    )
    parser.add_argument(
        "--return_instances",
        help="Return instances or tokens from parsect dataset",
        action="store_true",
    )
    parser.add_argument(
        "--bidirectional",
        help="Specify Whether the lstm is bidirectional or uni-directional",
        action="store_true",
    )
    parser.add_argument(
        "--combine_strategy",
        help="How do you want to combine the hidden dimensions of the two "
        "combinations",
    )
    args = parser.parse_args()
    config = {
        "EXP_NAME": args.exp_name,
        "DEVICE": args.device,
        "MAX_NUM_WORDS": args.max_num_words,
        "MAX_LENGTH": args.max_length,
        "DEBUG": args.debug,
        "DEBUG_DATASET_PROPORTION": args.debug_dataset_proportion,
        "BATCH_SIZE": args.bs,
        "EMBEDDING_TYPE": args.emb_type,
        "EMBEDDING_DIMENSION": args.emb_dim,
        "HIDDEN_DIMENSION": args.hidden_dim,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SAVE_EVERY": args.save_every,
        "LOG_TRAIN_METRICS_EVERY": args.log_train_metrics_every,
        "RETURN_INSTANCES": args.return_instances,
        "BIDIRECTIONAL": bool(args.bidirectional),
        "COMBINE_STRATEGY": args.combine_strategy,
        "ELMO_EMBEDDING_DIMENSION": 1024,
    }

    EXP_NAME = config["EXP_NAME"]
    DEVICE = config["DEVICE"]
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")
    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    EMBEDDING_TYPE = config["EMBEDDING_TYPE"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BATCH_SIZE = config["BATCH_SIZE"]
    HIDDEN_DIMENSION = config["HIDDEN_DIMENSION"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    SAVE_EVERY = config["SAVE_EVERY"]
    LOG_TRAIN_METRICS_EVERY = config["LOG_TRAIN_METRICS_EVERY"]
    RETURN_INSTANCES = config["RETURN_INSTANCES"]
    TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    ELMO_EMBEDDING_DIMENSION = config["ELMO_EMBEDDING_DIMENSION"]

    train_dataset = ParsectDataset(
        filename=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        return_instances=RETURN_INSTANCES,
    )

    validation_dataset = ParsectDataset(
        filename=SECT_LABEL_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        return_instances=RETURN_INSTANCES,
    )

    test_dataset = ParsectDataset(
        filename=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        return_instances=RETURN_INSTANCES,
    )

    VOCAB_SIZE = train_dataset.word_vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()
    embeddings = train_dataset.get_preloaded_word_embedding()
    embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)

    elmo_embedder = ElmoEmbedder(device=torch.device(DEVICE))
    elmo_lstm_encoder = ElmoLSTMEncoder(
        elmo_emb_dim=ELMO_EMBEDDING_DIMENSION,
        elmo_embedder=elmo_embedder,
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embeddings,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIMENSION,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        device=torch.device(DEVICE),
    )

    encoding_dim = 2 * HIDDEN_DIMENSION if BIDIRECTIONAL else HIDDEN_DIMENSION

    model = ElmoLSTMClassifier(
        elmo_lstm_encoder=elmo_lstm_encoder,
        encoding_dim=encoding_dim,
        num_classes=NUM_CLASSES,
        device=torch.device(DEVICE),
    )

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    metric = PrecisionRecallFMeasure(idx2labelname_mapping=train_dataset.idx2classname)

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
    config["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR
    config["VOCAB_SIZE"] = VOCAB_SIZE
    config["NUM_CLASSES"] = NUM_CLASSES
    with open(os.path.join(EXP_DIR_PATH, "config.json"), "w") as fp:
        json.dump(config, fp)
