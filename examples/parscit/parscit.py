from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
import sciwing.constants as constants
import torch
import torch.optim as optim
from sciwing.engine.engine import Engine
import argparse
import wasabi
import pathlib

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
    parser.add_argument("--model_save_dir", help="Model save directory", type=str)

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
        "glove_6B_200, glove_6B_300, parscit",
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
    parser.add_argument(
        "--sample_proportion",
        help="Sampling proportion of dataset for debugging",
        type=float,
    )

    args = parser.parse_args()
    msg_printer = wasabi.Printer()

    data_dir = pathlib.Path(DATA_DIR)
    train_filename = data_dir.joinpath("parscit.train")
    dev_filename = data_dir.joinpath("parscit.dev")
    test_filename = data_dir.joinpath("parscit.test")
    data_manager = SeqLabellingDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
    )
    embedder = WordEmbedder(embedding_type=args.emb_type, device=args.device)

    char_embedder = CharEmbedder(
        char_embedding_dimension=args.char_emb_dim,
        hidden_dimension=args.char_encoder_hidden_dim,
        datasets_manager=data_manager,
        device=args.device,
    )

    embedder = ConcatEmbedders([embedder, char_embedder])

    lstm2seqencoder = Lstm2SeqEncoder(
        embedder=embedder,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        combine_strategy=args.combine_strategy,
        rnn_bias=True,
        device=torch.device(args.device),
    )
    model = RnnSeqCrfTagger(
        rnn2seqencoder=lstm2seqencoder,
        encoding_dim=2 * args.hidden_dim
        if args.bidirectional and args.combine_strategy == "concat"
        else args.hidden_dim,
        device=torch.device(args.device),
        datasets_manager=data_manager,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    train_metric = TokenClassificationAccuracy(datasets_manager=data_manager)
    dev_metric = TokenClassificationAccuracy(datasets_manager=data_manager)
    test_metric = TokenClassificationAccuracy(datasets_manager=data_manager)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="max", factor=0.1, patience=2
    )

    engine = Engine(
        model=model,
        datasets_manager=data_manager,
        optimizer=optimizer,
        batch_size=args.bs,
        save_dir=args.model_save_dir,
        num_epochs=args.epochs,
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        save_every=args.save_every,
        log_train_metrics_every=args.log_train_metrics_every,
        device=torch.device(args.device),
        track_for_best="macro_fscore",
        lr_scheduler=scheduler,
        sample_proportion=args.sample_proportion,
    )

    engine.run()
