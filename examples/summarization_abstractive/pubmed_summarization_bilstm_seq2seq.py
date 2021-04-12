from sciwing.datasets.summarization.abstractive_text_summarization_dataset import (
    AbstractiveSummarizationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.lstm2seqdecoder import Lstm2SeqDecoder
from sciwing.models.simple_seq2seq import Seq2SeqModel
import pathlib
from sciwing.metrics.summarization_metrics import SummarizationMetrics
import torch.optim as optim
from sciwing.engine.engine import Engine
import argparse
import torch
import sciwing.constants as constants

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]


if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(description="Glove with LSTM encoder and decoder")

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)

    parser.add_argument(
        "--device", help="Specify the device where the model is run", type=str
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
    parser.add_argument(
        "--emb_type",
        help="The type of glove embedding you want. The allowed types are glove_6B_50, glove_6B_100, "
        "glove_6B_200, glove_6B_300, random",
    )
    parser.add_argument(
        "--hidden_dim", help="Hidden dimension of the LSTM network", type=int
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

    parser.add_argument(
        "--pred_max_length", help="Maximum length of prediction", type=int
    )

    parser.add_argument(
        "--exp_dir_path", help="Directory to store all experiment related information"
    )

    parser.add_argument(
        "--model_save_dir",
        help="Directory where the checkpoints during model training are stored.",
    )
    parser.add_argument(
        "--sample_proportion", help="Sample proportion for the dataset", type=float
    )

    args = parser.parse_args()

    DATA_PATH = pathlib.Path(DATA_DIR)
    train_file = DATA_PATH.joinpath("pubmedSeq2seq.train")
    dev_file = DATA_PATH.joinpath("pubmedSeq2seq.dev")
    test_file = DATA_PATH.joinpath("pubmedSeq2seq.test")

    data_manager = AbstractiveSummarizationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    vocab = data_manager.build_vocab()["tokens"]

    # # instantiate the elmo embedder
    # elmo_embedder = BowElmoEmbedder(layer_aggregation="sum", device=args.device)
    #
    # # instantiate the vanilla embedder
    # vanilla_embedder = WordEmbedder(embedding_type=args.emb_type, device=args.device)
    #
    # # concat the embeddings
    # embedder = ConcatEmbedders([vanilla_embedder, elmo_embedder])

    embedder = WordEmbedder(embedding_type=args.emb_type, device=args.device)

    encoder = Lstm2SeqEncoder(
        embedder=embedder,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        combine_strategy=args.combine_strategy,
        device=torch.device(args.device),
    )

    encoding_dim = (
        2 * args.hidden_dim
        if args.bidirectional and args.combine_strategy == "concat"
        else args.hidden_dim
    )

    decoder = Lstm2SeqDecoder(
        embedder=embedder,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        combine_strategy=args.combine_strategy,
        device=torch.device(args.device),
        max_length=args.pred_max_length,
        vocab=vocab,
    )

    model = Seq2SeqModel(
        rnn2seqencoder=encoder,
        rnn2seqdecoder=decoder,
        enc_hidden_dim=args.hidden_dim,
        datasets_manager=data_manager,
        device=args.device,
        bidirectional=args.bidirectional,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    train_metric = SummarizationMetrics(datasets_manager=data_manager)
    dev_metric = SummarizationMetrics(datasets_manager=data_manager)
    test_metric = SummarizationMetrics(datasets_manager=data_manager)

    engine = Engine(
        model=model,
        datasets_manager=data_manager,
        optimizer=optimizer,
        batch_size=args.bs,
        save_dir=args.model_save_dir,
        num_epochs=args.epochs,
        save_every=args.save_every,
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        log_train_metrics_every=args.log_train_metrics_every,
        device=torch.device(args.device),
        use_wandb=True,
        experiment_name=args.exp_name,
        experiment_hyperparams=vars(args),
        track_for_best="rouge_1",
        sample_proportion=args.sample_proportion,
    )

    engine.run()
