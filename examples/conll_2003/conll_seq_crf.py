from sciwing.datasets.seq_labeling.conll_dataset import CoNLLDatasetManager
import sciwing.constants as constants
import pathlib
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
import argparse
import wasabi
import torch
import torch.optim as optim
from sciwing.metrics.token_cls_accuracy import TokenClassificationAccuracy
from sciwing.engine.engine import Engine

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoNLL 2003 NER")

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
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
        "--exp_dir_path", help="Directory to store all experiment related information"
    )
    parser.add_argument(
        "--model_save_dir",
        help="Directory where the checkpoints during model training are stored.",
    )
    parser.add_argument(
        "--sample_proportion", help="Sample proportion of the dataset", type=float
    )

    parser.add_argument(
        "--num_layers", help="Number of layers in rnn2seq encoder", type=int
    )

    args = parser.parse_args()
    msg_printer = wasabi.Printer()

    data_dir = pathlib.Path(DATA_DIR)
    train_filename = data_dir.joinpath("eng.train")
    dev_filename = data_dir.joinpath("eng.testa")
    test_filename = data_dir.joinpath("eng.testb")

    data_manager = CoNLLDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
        column_names=["POS", "CHUNK", "NER"],
    )

    embedder = TrainableWordEmbedder(
        embedding_type=args.emb_type, datasets_manager=data_manager, device=args.device
    )

    char_embedder = CharEmbedder(
        char_embedding_dimension=args.char_emb_dim,
        hidden_dimension=args.char_encoder_hidden_dim,
        datasets_manager=data_manager,
        device=args.device,
    )
    embedder = ConcatEmbedders([embedder, char_embedder])

    lstm2seqencoder = Lstm2SeqEncoder(
        embedder=embedder,
        dropout_value=args.dropout,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        combine_strategy=args.combine_strategy,
        rnn_bias=True,
        device=torch.device(args.device),
        num_layers=args.num_layers,
    )
    model = RnnSeqCrfTagger(
        rnn2seqencoder=lstm2seqencoder,
        encoding_dim=2 * args.hidden_dim
        if args.bidirectional and args.combine_strategy == "concat"
        else args.hidden_dim,
        device=torch.device(args.device),
        tagging_type="BIOUL",
        datasets_manager=data_manager,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.reg)

    train_metric = TokenClassificationAccuracy(datasets_manager=data_manager)
    dev_metric = TokenClassificationAccuracy(datasets_manager=data_manager)
    test_metric = TokenClassificationAccuracy(datasets_manager=data_manager)

    engine = Engine(
        model=model,
        datasets_manager=data_manager,
        optimizer=optimizer,
        batch_size=args.bs,
        save_dir=args.model_save_dir,
        num_epochs=args.epochs,
        save_every=args.save_every,
        log_train_metrics_every=args.log_train_metrics_every,
        track_for_best="macro_fscore",
        device=torch.device(args.device),
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        use_wandb=True,
        experiment_name=args.exp_name,
        experiment_hyperparams=vars(args),
        sample_proportion=args.sample_proportion,
    )

    engine.run()
