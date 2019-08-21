from parsect.models.parscit_tagger import ParscitTagger
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.charlstm_encoder import CharLSTMEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.embedders.concat_embedders import ConcatEmbedders
from parsect.datasets.seq_labeling.parscit_dataset import ParscitDataset
from parsect.metrics.token_cls_accuracy import TokenClassificationAccuracy
from parsect.utils.common import write_nfold_parscit_train_test
from parsect.utils.common import merge_dictionaries_with_sum
from parsect.utils.common import write_cora_to_conll_file
from parsect.metrics.classification_metrics_utils import ClassificationMetricsUtils
from typing import Dict
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
import copy


FILES = constants.FILES
PATHS = constants.PATHS

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

    parser.add_argument("--dropout", help="Dropout value", type=float)
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
        "DROPOUT": args.dropout,
    }

    tp_counter = {}
    fp_counter = {}
    fn_counter = {}

    def setup_engine_once(
        config_dict: Dict[str, str],
        experiment_name: str,
        train_data_filepath: pathlib.Path,
        test_data_filepath: pathlib.Path,
    ):
        DEBUG = config_dict["DEBUG"]
        DEBUG_DATASET_PROPORTION = config_dict["DEBUG_DATASET_PROPORTION"]
        BATCH_SIZE = config_dict["BATCH_SIZE"]
        LEARNING_RATE = config_dict["LEARNING_RATE"]
        NUM_EPOCHS = config_dict["NUM_EPOCHS"]
        SAVE_EVERY = config_dict["SAVE_EVERY"]
        LOG_TRAIN_METRICS_EVERY = config_dict["LOG_TRAIN_METRICS_EVERY"]
        EMBEDDING_DIMENSION = config_dict["EMBEDDING_DIMENSION"]
        CHAR_EMBEDDING_DIMENSION = config_dict["CHAR_EMBEDDING_DIMENSION"]
        EMBEDDING_TYPE = config_dict["EMBEDDING_TYPE"]
        MAX_NUM_WORDS = config_dict["MAX_NUM_WORDS"]
        MAX_LENGTH = config_dict["MAX_LENGTH"]
        DEVICE = config_dict["DEVICE"]
        HIDDEN_DIM = config_dict["HIDDEN_DIM"]
        BIDIRECTIONAL = config_dict["BIDIRECTIONAL"]
        COMBINE_STRATEGY = config_dict["COMBINE_STRATEGY"]
        MAX_CHAR_LENGTH = config_dict["MAX_CHAR_LENGTH"]
        USE_CHAR_ENCODER = config_dict["USE_CHAR_ENCODER"]
        CHAR_ENCODER_HIDDEN_DIM = config_dict["CHAR_ENCODER_HIDDEN_DIM"]
        DROPOUT = config_dict["DROPOUT"]

        EXP_NAME = experiment_name
        EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)

        if not os.path.isdir(EXP_DIR_PATH):
            os.mkdir(EXP_DIR_PATH)

        MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, "checkpoints")

        if not os.path.isdir(MODEL_SAVE_DIR):
            os.mkdir(MODEL_SAVE_DIR)

        VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "vocab.json")
        CHAR_VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, "char_vocab.json")
        CAPITALIZATION_VOCAB_STORE_LOCATION = os.path.join(
            EXP_DIR_PATH, "capitalization_vocab.json"
        )
        CAPITALIZATION_EMBEDDING_DIMENSION = 10
        TENSORBOARD_LOGDIR = os.path.join(".", "runs", EXP_NAME)

        train_dataset = ParscitDataset(
            filename=str(train_data_filepath),
            dataset_type="train",
            max_num_words=MAX_NUM_WORDS,
            max_instance_length=MAX_LENGTH,
            max_char_length=MAX_CHAR_LENGTH,
            word_vocab_store_location=VOCAB_STORE_LOCATION,
            char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
            captialization_vocab_store_location=CAPITALIZATION_VOCAB_STORE_LOCATION,
            capitalization_emb_dim=CAPITALIZATION_EMBEDDING_DIMENSION,
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

        validation_dataset = ParscitDataset(
            filename=str(test_data_filepath),
            dataset_type="valid",
            max_num_words=MAX_NUM_WORDS,
            max_instance_length=MAX_LENGTH,
            max_char_length=MAX_CHAR_LENGTH,
            word_vocab_store_location=VOCAB_STORE_LOCATION,
            char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
            captialization_vocab_store_location=CAPITALIZATION_VOCAB_STORE_LOCATION,
            capitalization_emb_dim=CAPITALIZATION_EMBEDDING_DIMENSION,
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

        test_dataset = ParscitDataset(
            filename=str(test_data_filepath),
            dataset_type="test",
            max_num_words=MAX_NUM_WORDS,
            max_instance_length=MAX_LENGTH,
            max_char_length=MAX_CHAR_LENGTH,
            word_vocab_store_location=VOCAB_STORE_LOCATION,
            char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
            captialization_vocab_store_location=CAPITALIZATION_VOCAB_STORE_LOCATION,
            capitalization_emb_dim=CAPITALIZATION_EMBEDDING_DIMENSION,
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

        embedder = VanillaEmbedder(
            embedding=embedding, embedding_dim=EMBEDDING_DIMENSION
        )

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
            rnn_bias=True,
            device=torch.device(DEVICE),
        )
        model = ParscitTagger(
            rnn2seqencoder=lstm2seqencoder,
            num_classes=NUM_CLASSES,
            hid_dim=2 * HIDDEN_DIM
            if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
            else HIDDEN_DIM,
        )

        optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        metric = TokenClassificationAccuracy(
            idx2labelname_mapping=train_dataset.idx2classname
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", factor=0.1, patience=2
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
            lr_scheduler=scheduler,
        )

        config_dict["VOCAB_STORE_LOCATION"] = VOCAB_STORE_LOCATION
        config_dict["CHAR_VOCAB_STORE_LOCATION"] = CHAR_VOCAB_STORE_LOCATION
        config_dict["MODEL_SAVE_DIR"] = MODEL_SAVE_DIR
        config_dict["VOCAB_SIZE"] = VOCAB_SIZE
        config_dict["NUM_CLASSES"] = NUM_CLASSES

        with open(os.path.join(f"{EXP_DIR_PATH}", "config.json"), "w") as fp:
            json.dump(config_dict, fp)

        return engine

    train_conll_filepath = pathlib.Path(DATA_DIR, "parscit_train_conll.txt")
    test_conll_filepath = pathlib.Path(DATA_DIR, "parscit_test_conll.txt")

    for fold_num, each_success_indicator in enumerate(
        write_nfold_parscit_train_test(
            parscit_train_filepath=pathlib.Path(CORA_FILE),
            output_train_filepath=train_conll_filepath,
            output_test_filepath=test_conll_filepath,
            nsplits=10,
        )
    ):
        msg_printer.divider(f"RUNNING PARSCIT FOR FOLD {fold_num}")

        exp_name = config["EXP_NAME"]
        exp_name = f"{exp_name}_{fold_num}"

        engine = setup_engine_once(
            experiment_name=exp_name,
            config_dict=copy.deepcopy(config),
            train_data_filepath=train_conll_filepath,
            test_data_filepath=test_conll_filepath,
        )
        # generating one path for every fold run
        engine.run()

        fold_tp_counter = engine.test_metric_calc.tp_counter
        fold_fp_counter = engine.test_metric_calc.fp_counter
        fold_fn_counter = engine.test_metric_calc.fn_counter

        tp_counter = merge_dictionaries_with_sum(tp_counter, fold_tp_counter)
        fp_counter = merge_dictionaries_with_sum(fp_counter, fold_fp_counter)
        fn_counter = merge_dictionaries_with_sum(fn_counter, fold_fn_counter)

    parscit_classname2idx = ParscitDataset.get_classname2idx()
    idx2_classname = {
        idx: classname for classname, idx in parscit_classname2idx.items()
    }
    ignore_indices = [
        parscit_classname2idx["starting"],
        parscit_classname2idx["ending"],
        parscit_classname2idx["padding"],
    ]

    classification_metrics_utils = ClassificationMetricsUtils(
        idx2labelname_mapping=idx2_classname
    )
    table = classification_metrics_utils.generate_table_report_from_counters(
        tp_counter=tp_counter, fp_counter=fp_counter, fn_counter=fn_counter
    )

    EXP_DIR_PATH = pathlib.Path(OUTPUT_DIR, config["EXP_NAME"])

    if not EXP_DIR_PATH.is_dir():
        EXP_DIR_PATH.mkdir()

    test_results_filepath = EXP_DIR_PATH.joinpath("test_results.txt")

    with open(test_results_filepath, "w") as fp:
        fp.write(table)
        fp.write("\n")
    # run the model on the entire cora dataset
    cora_conll_filepath = pathlib.Path(DATA_DIR, "cora_conll_full.txt")
    write_cora_to_conll_file(cora_conll_filepath=cora_conll_filepath)

    engine = setup_engine_once(
        config_dict=copy.deepcopy(config),
        train_data_filepath=cora_conll_filepath,
        test_data_filepath=cora_conll_filepath,
        experiment_name=config["EXP_NAME"],
    )
    engine.run()
