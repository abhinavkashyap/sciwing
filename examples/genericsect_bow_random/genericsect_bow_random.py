from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.generic_sect_dataset import GenericSectDataset
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import sciwing.constants as constants
import os
import torch.optim as optim
from sciwing.engine.engine import Engine
import json
import argparse
import torch.nn as nn
import pathlib

FILES = constants.FILES
PATHS = constants.PATHS

GENERIC_SECTION_TRAIN_FILE = FILES["GENERIC_SECTION_TRAIN_FILE"]
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
CONFIGS_DIR = PATHS["CONFIGS_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Bag of words linear classifier. "
        "with initial random word embeddings"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument(
        "--max_num_words",
        help="Maximum number of words to be considered " "in the vocab",
        type=int,
    )
    parser.add_argument(
        "--max_length", help="Maximum length of every sentence", type=int
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
    parser.add_argument("--emb_dim", help="embedding dimension", type=int)
    parser.add_argument("--emb_type", help="embedding type", type=str)
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

    parser.add_argument(
        "--exp_dir_path", help="Directory to store all experiment related information"
    )
    parser.add_argument(
        "--model_save_dir",
        help="Directory where the checkpoints during model training are stored.",
    )
    parser.add_argument(
        "--vocab_store_location", help="File in which the vocab is stored"
    )

    args = parser.parse_args()
    config = {
        "EXP_NAME": args.exp_name,
        "MAX_NUM_WORDS": args.max_num_words,
        "MAX_LENGTH": args.max_length,
        "DEBUG": args.debug,
        "DEBUG_DATASET_PROPORTION": args.debug_dataset_proportion,
        "BATCH_SIZE": args.bs,
        "EMBEDDING_DIMENSION": args.emb_dim,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SAVE_EVERY": args.save_every,
        "LOG_TRAIN_METRICS_EVERY": args.log_train_metrics_every,
        "EMBEDDING_TYPE": args.emb_type,
        "EXP_DIR_PATH": args.exp_dir_path,
        "MODEL_SAVE_DIR": args.model_save_dir,
        "VOCAB_STORE_LOCATION": args.vocab_store_location,
    }

    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    VOCAB_STORE_LOCATION = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
    LEARNING_RATE = config["LEARNING_RATE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    SAVE_EVERY = config["SAVE_EVERY"]
    LOG_TRAIN_METRICS_EVERY = config["LOG_TRAIN_METRICS_EVERY"]
    EMBEDDING_TYPE = config["EMBEDDING_TYPE"]
    EXP_DIR_PATH = config["EXP_DIR_PATH"]
    EXP_DIR_PATH = pathlib.Path(EXP_DIR_PATH)
    EXP_NAME = config["EXP_NAME"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]

    GENERIC_SECTION_TRAIN_FILE = "genericSect.train.data"

    train_dataset = GenericSectDataset(
        filename=GENERIC_SECTION_TRAIN_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
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
        max_instance_length=MAX_LENGTH,
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
        max_instance_length=MAX_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
    )

    test_dataset_params = {
        "filename": GENERIC_SECTION_TRAIN_FILE,
        "dataset_type": "test",
        "max_num_words": MAX_NUM_WORDS,
        "max_instance_length": MAX_LENGTH,
        "word_vocab_store_location": VOCAB_STORE_LOCATION,
        "debug": DEBUG,
        "debug_dataset_proportion": DEBUG_DATASET_PROPORTION,
        "word_embedding_type": EMBEDDING_TYPE,
        "word_embedding_dimension": EMBEDDING_DIMENSION,
    }

    # saving the test dataset params
    # lets save the test dataset params for the experiment
    if not EXP_DIR_PATH.is_dir():
        EXP_DIR_PATH.mkdir()

    with open(os.path.join(EXP_DIR_PATH, "config.json"), "w") as fp:
        json.dump(config, fp)

    VOCAB_SIZE = train_dataset.word_vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()

    config["VOCAB_SIZE"] = VOCAB_SIZE
    config["NUM_CLASSES"] = NUM_CLASSES

    with open(os.path.join(EXP_DIR_PATH, "test_dataset_params.json"), "w") as fp:
        json.dump(test_dataset_params, fp)

    random_embeddings = train_dataset.word_vocab.load_embedding()
    embedding = nn.Embedding.from_pretrained(random_embeddings, freeze=False)

    embedder = VanillaEmbedder(embedding_dim=EMBEDDING_DIMENSION, embedding=embedding)
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        dropout_value=0.0,
        aggregation_type="sum",
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True,
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
        metric=metric,
        use_wandb=True,
        experiment_name=EXP_NAME,
        experiment_hyperparams=config,
        track_for_best="macro_fscore",
    )

    engine.run()
