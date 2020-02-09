from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
import sciwing.constants as constants
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
from sciwing.modules.embedders.word_embedder import WordEmbedder
import torch.optim as optim
from sciwing.engine.engine import Engine
import argparse
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file

    parser = argparse.ArgumentParser(
        description="Bag of words linear classifier. "
        "with initial random word embeddings"
    )
    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument("--bs", help="batch size", type=int)
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

    # initialize a bag of word emcoder
    encoder = BOW_Encoder(embedder=embedder)

    # Instantiate a simple classifier
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=embedder.get_embedding_dimension(),
        classification_layer_bias=True,
        num_classes=data_manager.num_labels["label"],
        datasets_manager=data_manager,
    )

    # you get to use any optimizer from Pytorch
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    # Instantiate the PrecisionRecallFMeasure
    train_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)
    dev_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)
    test_metric = PrecisionRecallFMeasure(datasets_manager=data_manager)

    # Get the engine worked up
    engine = Engine(
        model=model,
        datasets_manager=data_manager,
        optimizer=optimizer,
        batch_size=args.bs,
        save_dir=args.model_save_dir,
        num_epochs=args.epochs,
        save_every=args.save_every,
        log_train_metrics_every=args.log_train_metrics_every,
        train_metric=train_metric,
        validation_metric=dev_metric,
        test_metric=test_metric,
        use_wandb=True,
        experiment_name=args.exp_name,
        experiment_hyperparams=vars(args),
        track_for_best="macro_fscore",
        sample_proportion=0.01,
    )

    # Run the engine
    engine.run()
