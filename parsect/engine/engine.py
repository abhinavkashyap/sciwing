from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from wasabi import Printer
from typing import Iterator, Callable, Any, List
from parsect.meters.loss_meter import LossMeter
import os
from tensorboardX import SummaryWriter
from parsect.metrics.precision_recall_fmeasure import PrecisionRecallFMeasure
import numpy as np
import time
import json_logging
import logging
from torch.utils.data._utils.collate import default_collate
import torch
from parsect.utils.tensor import move_to_device


class Engine:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: optim,
        batch_size: int,
        save_dir: str,
        num_epochs: int,
        save_every: int,
        log_train_metrics_every: int,
        tensorboard_logdir: str = None,
        metric: str = "accuracy",
        track_for_best: str = "loss",
        collate_fn: Callable[[List[Any]], List[Any]] = default_collate,
        device=torch.device("cpu"),
    ):
        """
        This orchestrates the whole model training. The supervised machine learning
        that needs to be used
        :param model: type: nnn.Module
        Any pytorch model that takes in inputs
        :param train_dataset: type: torch.utils.data.Dataset
        training dataset, that can be iterated and follows the Dataset conventions of PyTorch
        :param validation_dataset: type: torch.utils.data.Dataset
        Validation dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param test_dataset: type: torch.utils.Dataset
        Test Dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param optimizer: torch.optim
        Any optimizer that belongs to the torch.optim module
        :param batch_size: type: int
        Batch size for the train, validation and test dataset
        :param save_dir: type: str
        The full location where the intermediate results are stored
        :param num_epochs: type: int
        Number of epochs to run training
        :param save_every: type: int
        The model state will be save every `save_every` num of epochs
        :param log_train_metrics_every
        The train metrics will logged every `log_train_metrics_every` iterations
        :param tensorboard_logdir: type: str
        pass in the directory where tensorboard logs are stored
        By default it will be put in a directory called run/ in the same directory as the
        script using Engine
        :param track_for_best: type: str
        This will be tracked in order to determine the best model parameters
        to save
        :param collate_fn: type: Callable[[List[Any]], List[Any]]
        - The collate function that gets used with the data loader.
        -The collate function provides the logic to group a batch of instances
        - There are more details in `torch.utils.data.DataLoader`.
        - If None, pytorchs default collate function will be used
        :param device: torch.device
        Torch device to which the model and the tensors will be converted
        """

        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.msg_printer = Printer()
        self.save_every = save_every
        self.log_train_metrics_every = log_train_metrics_every
        self.tensorboard_logdir = tensorboard_logdir
        self.metric = metric
        self.summaryWriter = SummaryWriter(log_dir=tensorboard_logdir)
        self.track_for_best = track_for_best
        self.collate_fn = collate_fn
        self.device = device
        self.best_track_value = None
        self.set_best_track_value(self.best_track_value)

        self.num_workers = 0

        self.labelname2idx_mapping = self.train_dataset.get_classname2idx()
        self.idx2labelname_mapping = {
            idx: label_name for label_name, idx in self.labelname2idx_mapping.items()
        }

        # get the data loader
        # TODO: For now we randomly sample the dataset to obtain instances, we can have different
        #       sampling strategies. For one, there are BucketIterators, that bucket different
        #       isntances of the same length together

        self.model.to(self.device)

        self.train_loader = self.get_loader(self.train_dataset)
        self.validation_loader = self.get_loader(self.validation_dataset)
        self.test_loader = self.get_loader(self.test_dataset)

        # refresh the iters at the beginning of every epoch
        self.train_iter = None
        self.validation_iter = None
        self.test_iter = None

        # initializing loss meters
        self.train_loss_meter = LossMeter()
        self.validation_loss_meter = LossMeter()

        # get metric calculators
        self.train_metric_calc, self.validation_metric_calc, self.test_metric_calc = (
            self.get_metric_calculators()
        )

        self.msg_printer.divider("ENGINE STARTING")
        self.msg_printer.info(
            "Number of training examples {0}".format(len(self.train_dataset))
        )
        self.msg_printer.info(
            "Number of validation examples {0}".format(len(self.validation_dataset))
        )
        self.msg_printer.info(
            "Number of test examples {0}".format(len(self.test_dataset))
        )
        time.sleep(3)

        # get the loggers ready
        # TODO - use loguru - seems simple and has many advantages
        json_logging.ENABLE_JSON_LOGGING = True
        json_logging.init()
        self.train_log_filename = os.path.join(self.save_dir, "train.log")
        self.validation_log_filename = os.path.join(self.save_dir, "validation.log")
        self.test_log_filename = os.path.join(self.save_dir, "test.log")

        self.train_logger = logging.getLogger("train-logger")
        self.validation_logger = logging.getLogger("validation-logger")
        self.test_logger = logging.getLogger("test-logger")

        self.train_logger.setLevel(logging.DEBUG)
        self.train_logger.addHandler(logging.FileHandler(self.train_log_filename))

        self.validation_logger.setLevel(logging.DEBUG)
        self.validation_logger.addHandler(
            logging.FileHandler(self.validation_log_filename)
        )

        self.test_logger.setLevel(logging.DEBUG)
        self.test_logger.addHandler(logging.FileHandler(self.test_log_filename))

    def get_loader(self, dataset: Dataset) -> DataLoader:
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        return loader

    def is_best_lower(self, current_best=None):
        return True if current_best < self.best_track_value else False

    def set_best_track_value(self, current_best=None):
        if self.track_for_best == "loss":
            self.best_track_value = np.inf if current_best is None else current_best

    def run(self):
        """
        Run the engine
        :return:
        """
        for epoch_num in range(self.num_epochs):
            self.train_epoch(epoch_num)
            self.validation_epoch(epoch_num)

        self.test_epoch(epoch_num)

    def train_epoch(self, epoch_num: int):
        """
        Run the training for one epoch
        :param epoch_num: type: int
        The current epoch number
        """

        # refresh everything necessary before training begins
        num_iterations = 0
        train_iter = self.get_iter(self.train_loader)
        self.model.train()
        self.train_loss_meter.reset()
        self.train_metric_calc.reset()

        self.msg_printer.info("starting training epoch")
        while True:
            try:
                # N*T, N * 1, N * 1
                iter_dict = next(train_iter)
                iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
                labels = iter_dict["label"]
                batch_size = labels.size()[0]
                labels = labels.squeeze(1)

                model_forward_out = self.model(
                    iter_dict, is_training=True, is_validation=False, is_test=False
                )
                self.train_metric_calc.calc_metric(
                    model_forward_out["normalized_probs"].cpu(), labels.cpu()
                )

                try:
                    self.optimizer.zero_grad()
                    loss = model_forward_out["loss"]
                    loss.backward()
                    self.optimizer.step()
                    self.train_loss_meter.add_loss(loss.item(), batch_size)

                except KeyError:
                    self.msg_printer.fail(
                        "The model output dictionary does not have "
                        "a key called loss. Please check to have "
                        "loss in the model output"
                    )
                num_iterations += 1
                if (num_iterations + 1) % self.log_train_metrics_every == 0:
                    metrics = self.train_metric_calc.report_metrics()
                    precision_recall_fmeasure = self.train_metric_calc.get_metric()
                    print(metrics)
                    self.train_logger.info(
                        "Training Metrics at iteration {0} - {1}".format(
                            num_iterations + 1, precision_recall_fmeasure
                        )
                    )
            except StopIteration:
                self.train_epoch_end(epoch_num)
                break

    def train_epoch_end(self, epoch_num: int):
        """

        :param epoch_num: type: int
        The epoch number that just ended
        """

        self.msg_printer.divider("Training end @ Epoch {0}".format(epoch_num + 1))
        average_loss = self.train_loss_meter.get_average()
        self.msg_printer.text("Average Loss: {0}".format(average_loss))
        self.train_logger.info(
            "Average loss @ Epoch {0} - {1}".format(epoch_num + 1, average_loss)
        )

        # save the model after every `self.save_every` epochs
        if (epoch_num + 1) % self.save_every == 0:
            torch.save(
                {
                    "epoch_num": epoch_num,
                    "optimizer_state": self.optimizer.state_dict(),
                    "model_state": self.model.state_dict(),
                    "loss": average_loss,
                },
                os.path.join(self.save_dir, "model_epoch_{0}.pt".format(epoch_num + 1)),
            )

        # log loss to tensor board
        self.summaryWriter.add_scalars(
            "train_validation_loss",
            {"train_loss": average_loss or np.inf},
            epoch_num + 1,
        )

    def validation_epoch(self, epoch_num: int):
        """
        Run the validation
        """

        self.model.eval()
        valid_iter = iter(self.validation_loader)
        self.validation_loss_meter.reset()
        self.validation_metric_calc.reset()

        while True:
            try:
                iter_dict = next(valid_iter)
                iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
                labels = iter_dict["label"]
                batch_size = labels.size(0)
                labels = labels.squeeze(1)

                with torch.no_grad():
                    model_forward_out = self.model(
                        iter_dict, is_training=False, is_validation=True, is_test=False
                    )
                loss = model_forward_out["loss"]
                self.validation_loss_meter.add_loss(loss, batch_size)
                self.validation_metric_calc.calc_metric(
                    predicted_probs=model_forward_out["normalized_probs"].cpu(),
                    labels=labels.cpu(),
                )
            except StopIteration:
                self.validation_epoch_end(epoch_num)
                break

    def validation_epoch_end(self, epoch_num: int):

        self.msg_printer.divider("Validation @ Epoch {0}".format(epoch_num))

        metrics = self.validation_metric_calc.report_metrics()
        precision_recall_fmeasure = self.validation_metric_calc.get_metric()
        average_loss = self.validation_loss_meter.get_average()
        print(metrics)

        self.msg_printer.text("Average Loss: {0}".format(average_loss))

        self.validation_logger.info(
            "Validation Metrics @ Epoch {0} - {1}".format(
                epoch_num + 1, precision_recall_fmeasure
            )
        )
        self.validation_logger.info(
            "Validation Loss @ Epoch {0} - {1}".format(epoch_num + 1, average_loss)
        )

        self.summaryWriter.add_scalars(
            "train_validation_loss",
            {"validation_loss": average_loss or np.inf},
            epoch_num + 1,
        )

        if self.track_for_best == "loss":
            is_best = self.is_best_lower(average_loss)
            self.msg_printer.good(f"Found best model @ epoch {epoch_num + 1}")
            if is_best:
                self.set_best_track_value(average_loss)
                torch.save(
                    {
                        "epoch_num": epoch_num,
                        "optimizer_state": self.optimizer.state_dict(),
                        "model_state": self.model.state_dict(),
                        "loss": average_loss,
                    },
                    os.path.join(self.save_dir, "best_model.pt"),
                )

    def test_epoch(self, epoch_num: int):
        self.msg_printer.divider("Running on test batch")
        self.load_model_from_file(os.path.join(self.save_dir, "best_model.pt"))
        self.model.eval()
        test_iter = iter(self.test_loader)
        while True:
            try:
                iter_dict = next(test_iter)
                iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
                labels = iter_dict["label"]
                labels = labels.squeeze(1)

                with torch.no_grad():
                    model_forward_out = self.model(
                        iter_dict, is_training=False, is_validation=False, is_test=True
                    )
                self.test_metric_calc.calc_metric(
                    predicted_probs=model_forward_out["normalized_probs"].cpu(),
                    labels=labels.cpu(),
                )
            except StopIteration:
                self.test_epoch_end(epoch_num)
                break

    def test_epoch_end(self, epoch_num: int):
        metrics = self.test_metric_calc.report_metrics()
        precision_recall_fmeasure = self.test_metric_calc.get_metric()
        self.msg_printer.divider("Test @ Epoch {0}".format(epoch_num + 1))
        print(metrics)
        self.test_logger.info(
            "Test Metrics @ Epoch {0} - {1}".format(
                epoch_num + 1, precision_recall_fmeasure
            )
        )

    def get_train_dataset(self):
        return self.train_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset

    @staticmethod
    def get_iter(loader: DataLoader) -> Iterator:
        """
        The iterators return the next batch of instances
        when they are called. This will be useful
        for going over the dataset in a batch wise manner
        :return:
        """
        iterator = iter(loader)
        return iterator

    def load_model_from_file(self, filename: str):
        """
        This loads the pretrained model from file
        :param filename: type: str
        The filename where the model state is stored
        The model is saved during training. Look at the method `train_epoch_end` for
        more details.
        """
        self.msg_printer.divider("LOADING MODEL FROM FILE")
        with self.msg_printer.loading(
            "Loading Pytorch Model from file {0}".format(filename)
        ):
            model_chkpoint = torch.load(filename)

        self.msg_printer.good("Finished Loading the Model")

        model_state = model_chkpoint["model_state"]
        self.model.load_state_dict(model_state)

    def get_metric_calculators(self):
        train_calculator = None
        validation_calculator = None
        test_calculator = None

        if self.metric == "accuracy":
            train_calculator = PrecisionRecallFMeasure(
                idx2labelname_mapping=self.idx2labelname_mapping
            )
            validation_calculator = PrecisionRecallFMeasure(
                idx2labelname_mapping=self.idx2labelname_mapping
            )
            test_calculator = PrecisionRecallFMeasure(
                idx2labelname_mapping=self.idx2labelname_mapping
            )

        return train_calculator, validation_calculator, test_calculator


if __name__ == "__main__":  # pragma: no cover
    import parsect.constants as constants
    from parsect.datasets.parsect_dataset import ParsectDataset
    from parsect.modules.bow_encoder import BOW_Encoder
    from parsect.models.simpleclassifier import SimpleClassifier
    from torch.nn import Embedding

    FILES = constants.FILES
    SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]

    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 50
    vocab_store_location = os.path.join(".", "vocab.json")
    DEBUG = True

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="train",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="valid",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
    )

    BATCH_SIZE = 1
    NUM_TOKENS = 3
    EMB_DIM = 300
    NUM_CLASSES = train_dataset.get_num_classes()
    VOCAB_SIZE = MAX_NUM_WORDS + len(train_dataset.vocab.special_vocab)
    embedding = Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    labels = torch.LongTensor([1])

    encoder = BOW_Encoder(
        emb_dim=EMB_DIM, embedding=embedding, dropout_value=0, aggregation_type="sum"
    )
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        classification_layer_bias=False,
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    engine = Engine(
        model,
        train_dataset,
        validation_dataset,
        test_dataset,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        save_dir=os.path.join("."),
        num_epochs=1,
        save_every=1,
    )

    # engine.train_epoch_end(0)
    #     # engine.load_model_from_file(
    #     #     os.path.join(engine.save_dir, 'model_epoch_{0}.pt'.format(1))
    #     # )
    engine.run()
    # clean up
    os.remove("./vocab.json")
    os.remove(os.path.join(engine.save_dir, "model_epoch_{0}.pt".format(1)))
