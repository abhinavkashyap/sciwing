from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bert_embedder import BertEmbedder
from sciwing.modules.bow_encoder import BOW_Encoder
from sciwing.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import sciwing.constants as constants
import torch.optim as optim
from sciwing.engine.engine import Engine
import pathlib
import argparse
import torch

PATHS = constants.PATHS
DATA_DIR = PATHS["DATA_DIR"]

if __name__ == "__main__":
    # read the hyperparams from config file
    parser = argparse.ArgumentParser(
        description="Bag of words linear classifier. "
        "with initial elmo word embeddings"
    )

    parser.add_argument("--exp_name", help="Specify an experiment name", type=str)
    parser.add_argument("--device", help="specify the device to run on.", type=str)
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
        "--bert_type",
        help="Specify the bert model to be used. One of bert-base-uncased, bert-base-cased, "
        "bert-large-uncased, bert-large-cased can be used",
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
    train_file = DATA_PATH.joinpath("sectLabel.train")
    dev_file = DATA_PATH.joinpath("sectLabel.dev")
    test_file = DATA_PATH.joinpath("sectLabel.test")

    data_manager = TextClassificationDatasetManager(
        train_filename=str(train_file),
        dev_filename=str(dev_file),
        test_filename=str(test_file),
    )

    embedder = BertEmbedder(
        dropout_value=0.0,
        aggregation_type="average",
        bert_type=args.bert_type,
        device=torch.device(args.device),
    )

    encoder = BOW_Encoder(
        embedder=embedder, aggregation_type="average", device=args.device
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=768,
        num_classes=23,
        classification_layer_bias=True,
        datasets_manager=data_manager,
        device=args.device,
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
