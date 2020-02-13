from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.modules.embedders.word_embedder import WordEmbedder
import sciwing.constants as constants
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import torch.optim as optim
from sciwing.engine.engine import Engine
import argparse
import torch
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="LSTM encoder with linear classifier"
        "with initial random word embeddings"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument(
        "--device", help="Adding which device to run the experiment on", type=str
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

    # saving the test dataset params
    # lets save the test dataset params for the experiment

    DATA_PATH = pathlib.Path(DATA_DIR)
    train_file = DATA_PATH.joinpath("sectLabel.train")
    dev_file = DATA_PATH.joinpath("sectLabel.dev")
    test_file = DATA_PATH.joinpath("sectLabel.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    embedder = WordEmbedder(embedding_type=args.emb_type)
    encoder = LSTM2VecEncoder(
        embedder=embedder,
        hidden_dim=args.hidden_dim,
        combine_strategy=args.combine_strategy,
        bidirectional=args.bidirectional,
        device=torch.device(args.device),
    )

    classiier_encoding_dim = (
        2 * args.hidden_dim if args.bidirectional else args.hidden_dim
    )
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=classiier_encoding_dim,
        num_classes=23,
        classification_layer_bias=True,
        datasets_manager=data_manager,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    train_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)
    dev_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)
    test_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)

    engine = Engine(
        model=model,
        datasets_manager=data_manager,
        optimizer=optimizer,
        batch_size=args.bs,
        save_dir=args.model_save_dir,
        num_epochs=args.epochs,
        save_every=args.save_every,
        log_train_metrics_every=args.log_train_metrics_every,
        device=torch.device(args.device),
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        use_wandb=True,
        experiment_name=args.exp_name,
        experiment_hyperparams=vars(args),
        track_for_best="macro_fscore",
        sample_proportion=args.sample_proportion,
    )

    engine.run()
