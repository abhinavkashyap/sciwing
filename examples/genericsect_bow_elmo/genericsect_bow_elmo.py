from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import sciwing.constants as constants
import torch.optim as optim
from sciwing.engine.engine import Engine
import argparse
import torch
import re
import pathlib

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Bag of words linear classifier. "
        "with initial elmo word embeddings"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument("--device", help="device to run the models", type=str)
    parser.add_argument(
        "--layer_aggregation", help="Layer aggregation strategy", type=str
    )

    parser.add_argument(
        "--word_aggregation", help="word aggregation strategy", type=str
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
        "--exp_dir_path", help="Directory to store all experiment related information"
    )
    parser.add_argument(
        "--model_save_dir",
        help="Directory where the checkpoints during model training are stored.",
    )
    parser.add_argument("--sample_proportion", help="Sample data size", type=float)

    args = parser.parse_args()

    DATA_DIR = pathlib.Path(DATA_DIR)
    train_filename = DATA_DIR.joinpath("genericSect.train")
    dev_filename = DATA_DIR.joinpath("genericSect.dev")
    test_filename = DATA_DIR.joinpath("genericSect.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=test_filename,
    )

    embedder = BowElmoEmbedder(
        layer_aggregation=args.layer_aggregation,
        cuda_device_id=0 if re.match("cuda", args.device) else -1,
    )

    encoder = BOW_Encoder(aggregation_type=args.word_aggregation, embedder=embedder)

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=1024,
        num_classes=12,
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
    )

    engine.run()
