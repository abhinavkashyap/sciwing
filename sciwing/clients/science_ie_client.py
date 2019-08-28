from sciwing.models.science_ie_tagger import ScienceIETagger
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.charlstm_encoder import CharLSTMEncoder
from sciwing.modules.embedders.vanilla_embedder import VanillaEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.datasets.seq_labeling.science_ie_dataset import ScienceIEDataset
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from sciwing.utils.science_ie_data_utils import ScienceIEDataUtils
import sciwing.constants as constants
from allennlp.modules.conditional_random_field import allowed_transitions
import os
import torch
import torch.optim as optim
from sciwing.engine.engine import Engine
import json
import argparse
import pathlib
import torch.nn as nn
import wasabi


FILES = constants.FILES
PATHS = constants.PATHS

SCIENCE_IE_TRAIN_FOLDER = FILES["SCIENCE_IE_TRAIN_FOLDER"]
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]
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

    parser.add_argument("--reg", help="Regularization strength", type=float)
    parser.add_argument(
        "--dropout", help="Dropout added to multiple layer lstm", type=float
    )
    parser.add_argument(
        "--seq_num_layers", help="Number of layers in the Seq2Seq encoder", type=int
    )

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
        "REGULARIZATION_STRENGTH": args.reg,
        "DROPOUT": args.dropout,
        "NUM_LAYERS": args.seq_num_layers,
    }

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
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    DEVICE = config["DEVICE"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    BIDIRECTIONAL = config["BIDIRECTIONAL"]
    COMBINE_STRATEGY = config["COMBINE_STRATEGY"]
    MAX_CHAR_LENGTH = config["MAX_CHAR_LENGTH"]
    USE_CHAR_ENCODER = config["USE_CHAR_ENCODER"]
    CHAR_ENCODER_HIDDEN_DIM = config["CHAR_ENCODER_HIDDEN_DIM"]
    REGULARIZATION_STRENGTH = config["REGULARIZATION_STRENGTH"]
    DROPOUT = config["DROPOUT"]
    EXP_NAME = config["EXP_NAME"]
    NUM_LAYERS = config["NUM_LAYERS"]
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")

    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
    CHAR_VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "char_vocab.json")
    TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)

    train_dataset = ScienceIEDataset(
        filename=pathlib.Path(DATA_DIR, "train_science_ie_conll.txt"),
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        char_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        word_start_token="<SOS>",
        word_end_token="<EOS>",
        word_pad_token="<PAD>",
        word_unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    validation_dataset = ScienceIEDataset(
        filename=pathlib.Path(DATA_DIR, "dev_science_ie_conll.txt"),
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        char_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        word_start_token="<SOS>",
        word_end_token="<EOS>",
        word_pad_token="<PAD>",
        word_unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    test_dataset = ScienceIEDataset(
        filename=pathlib.Path(DATA_DIR, "dev_science_ie_conll.txt"),
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_instance_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        char_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        word_start_token="<SOS>",
        word_end_token="<EOS>",
        word_pad_token="<PAD>",
        word_unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    train_dataset.print_stats()
    validation_dataset.print_stats()
    test_dataset.print_stats()

    VOCAB_SIZE = train_dataset.word_vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()
    embedding = train_dataset.word_vocab.load_embedding()
    embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
    char_embedding = train_dataset.char_vocab.load_embedding()
    char_embedding = nn.Embedding.from_pretrained(char_embedding, freeze=False)

    classnames2idx = train_dataset.classnames2idx
    idx2classnames = {idx: classname for classname, idx in classnames2idx.items()}

    task_idx2classnames = {
        idx: classname
        for idx, classname in idx2classnames.items()
        if idx in range(0, 8)
    }
    process_idx2classnames = {
        idx - 8: classname
        for idx, classname in idx2classnames.items()
        if idx in range(8, 16)
    }
    material_idx2classnames = {
        idx - 16: classname
        for idx, classname in idx2classnames.items()
        if idx in range(16, 24)
    }

    task_constraints = allowed_transitions(
        constraint_type="BIOUL", labels=task_idx2classnames
    )
    process_constraints = allowed_transitions(
        constraint_type="BIOUL", labels=process_idx2classnames
    )
    material_constraints = allowed_transitions(
        constraint_type="BIOUL", labels=material_idx2classnames
    )

    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=EMBEDDING_DIMENSION)

    if USE_CHAR_ENCODER:
        char_embedder = VanillaEmbedder(
            embedding=char_embedding, embedding_dim=CHAR_EMBEDDING_DIMENSION
        )
        char_encoder = CharLSTMEncoder(
            char_emb_dim=CHAR_EMBEDDING_DIMENSION,
            char_embedder=char_embedder,
            bidirectional=True,
            hidden_dim=CHAR_ENCODER_HIDDEN_DIM,
            combine_strategy="concat",
            device=torch.device(DEVICE),
        )
        embedder = ConcatEmbedders([embedder, char_encoder])
        EMBEDDING_DIMENSION += 2 * CHAR_ENCODER_HIDDEN_DIM

    lstm2seqencoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        dropout_value=DROPOUT,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        num_layers=NUM_LAYERS,
        rnn_bias=True,
        device=torch.device(DEVICE),
    )
    model = ScienceIETagger(
        rnn2seqencoder=lstm2seqencoder,
        num_classes=NUM_CLASSES,
        hid_dim=2 * HIDDEN_DIM
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIM,
        task_constraints=task_constraints,
        process_constraints=process_constraints,
        material_constraints=material_constraints,
        device=torch.device(DEVICE),
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=REGULARIZATION_STRENGTH,
    )

    metric = TokenClassificationAccuracy(
        idx2labelname_mapping=train_dataset.idx2classnames
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
        track_for_best="macro_fscore",
    )

    engine.run()

    config["VOCAB_STORE_LOCATION"] = VOCAB_STORE_LOCATION
    config["CHAR_VOCAB_STORE_LOCATION"] = CHAR_VOCAB_STORE_LOCATION
    config["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR
    config["VOCAB_SIZE"] = VOCAB_SIZE
    config["NUM_CLASSES"] = NUM_CLASSES

    with open(os.path.join(f"{EXP_DIR_PATH}", "config.json"), "w") as fp:
        json.dump(config, fp)
