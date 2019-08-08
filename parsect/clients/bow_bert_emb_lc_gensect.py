from parsect.modules.embedders.bert_embedder import BertEmbedder
from parsect.modules.bow_encoder import BOW_Encoder
from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.classification.generic_sect_dataset import GenericSectDataset
import parsect.constants as constants
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import os

import torch.optim as optim
from parsect.engine.engine import Engine
import json
import argparse
import torch

FILES = constants.FILES
PATHS = constants.PATHS

GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Bag of words linear classifier. "
        "with initial elmo word embeddings"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument("--device", help="specify the device to run on.", type=str)
    parser.add_argument(
        "--max_num_words",
        help="Maximum number of words to be considered " "in the vocab",
        type=int,
    )

    parser.add_argument("--max_length", help="max length of sentences", type=int)
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
        "--return_instances",
        help="Set this if the dataset has to return instances",
        action="store_true",
    )
    parser.add_argument(
        "--bert_type",
        help="Specify the bert model to be used. One of bert-base-uncased, bert-base-cased, "
        "bert-large-uncased, bert-large-cased can be used",
    )
    args = parser.parse_args()

    config = {
        "EXP_NAME": args.exp_name,
        "DEVICE": args.device,
        "DEBUG": args.debug,
        "DEBUG_DATASET_PROPORTION": args.debug_dataset_proportion,
        "BATCH_SIZE": args.bs,
        "EMBEDDING_DIMENSION": args.emb_dim,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SAVE_EVERY": args.save_every,
        "LOG_TRAIN_METRICS_EVERY": args.log_train_metrics_every,
        "EMBEDDING_TYPE": args.emb_type,
        "RETURN_INSTANCES": args.return_instances,
        "BERT_TYPE": args.bert_type,
        "MAX_LENGTH": args.max_length,
        "MAX_NUM_WORDS": args.max_num_words,
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
    RETURN_INSTANCES = config["RETURN_INSTANCES"]
    TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)
    BERT_TYPE = config["BERT_TYPE"]
    DEVICE = config["DEVICE"]
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]

    train_dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    validation_dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    test_dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    VOCAB_SIZE = train_dataset.vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()
    random_embeddings = train_dataset.get_preloaded_word_embedding()

    embedder = BertEmbedder(
        emb_dim=EMBEDDING_DIMENSION,
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=BERT_TYPE,
        device=torch.device(DEVICE),
    )

    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION, embedder=embedder, aggregation_type="average"
    )
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    metric = PrecisionRecallFMeasure(idx2labelname_mapping=train_dataset.idx2label)

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
