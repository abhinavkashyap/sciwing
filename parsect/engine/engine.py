from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from wasabi import Printer
from typing import Iterator, Callable, Any, List, Optional
from parsect.meters.loss_meter import LossMeter
import os
from tensorboardX import SummaryWriter
from parsect.metrics.BaseMetric import BaseMetric
import numpy as np
import time
import logging
from torch.utils.data._utils.collate import default_collate
import torch
from parsect.utils.tensor_utils import move_to_device
from copy import deepcopy
from parsect.utils.class_nursery import ClassNursery
import logzero


class Engine(ClassNursery):
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
        metric: BaseMetric,
        tensorboard_logdir: str = None,
        track_for_best: str = "loss",
        collate_fn: Callable[[List[Any]], List[Any]] = default_collate,
        device=torch.device("cpu"),
        gradient_norm_clip_value: Optional[float] = 5.0,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):

        if isinstance(device, str):
            device = torch.device(device)

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
        self.gradient_norm_clip_value = gradient_norm_clip_value
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_is_plateau = isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        self.num_workers = 0

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
        self.train_metric_calc = deepcopy(metric)
        self.validation_metric_calc = deepcopy(metric)
        self.test_metric_calc = deepcopy(metric)

        self.msg_printer.divider("ENGINE STARTING")
        self.msg_printer.info(f"Number of training examples {len(self.train_dataset)}")
        self.msg_printer.info(
            f"Number of validation examples {len(self.validation_dataset)}"
        )
        self.msg_printer.info(
            f"Number of test examples {0}".format(len(self.test_dataset))
        )
        time.sleep(3)

        # get the loggers ready
        self.train_log_filename = os.path.join(self.save_dir, "train.log")
        self.validation_log_filename = os.path.join(self.save_dir, "validation.log")
        self.test_log_filename = os.path.join(self.save_dir, "test.log")

        self.train_logger = logzero.setup_logger(
            name="train-logger", logfile=self.train_log_filename, level=logging.INFO
        )
        self.validation_logger = logzero.setup_logger(
            name="valid-logger",
            logfile=self.validation_log_filename,
            level=logging.INFO,
        )
        self.test_logger = logzero.setup_logger(
            name="test-logger", logfile=self.test_log_filename, level=logging.INFO
        )

        if self.lr_scheduler_is_plateau:
            if self.best_track_value == "loss" and self.lr_scheduler.mode == "max":
                self.msg_printer.warn(
                    "You are optimizing loss and lr schedule mode is max instead of min"
                )
            if self.best_track_value == "macro-f1" and self.lr_scheduler.mode == "min":
                self.msg_printer.warn(
                    f"You are optimizing for macro-f1 and lr scheduler mode is min instead of max"
                )
            if self.best_track_value == "micro-f1" and self.lr_scheduler.mode == "min":
                self.msg_printer.warn(
                    f"You are optimizing for micro-f1 and lr scheduler mode is min instead of max"
                )

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

    def is_best_higher(self, current_best=None):
        return True if current_best >= self.best_track_value else False

    def set_best_track_value(self, current_best=None):
        if self.track_for_best == "loss":
            self.best_track_value = np.inf if current_best is None else current_best
        elif self.track_for_best == "macro-f1":
            self.best_track_value = 0 if current_best is None else current_best
        elif self.track_for_best == "micro-f1":
            self.best_track_value = 0 if current_best is None else current_best

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
        self.train_loss_meter.reset()
        self.train_metric_calc.reset()
        self.model.train()

        self.msg_printer.info("starting training epoch")
        while True:
            try:
                # N*T, N * 1, N * 1
                iter_dict = next(train_iter)
                iter_dict = move_to_device(obj=iter_dict, cuda_device=self.device)
                labels = iter_dict["label"]
                batch_size = labels.size()[0]

                model_forward_out = self.model(
                    iter_dict, is_training=True, is_validation=False, is_test=False
                )
                self.train_metric_calc.calc_metric(
                    iter_dict=iter_dict, model_forward_dict=model_forward_out
                )

                try:
                    self.optimizer.zero_grad()
                    loss = model_forward_out["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.gradient_norm_clip_value
                    )
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
                    print(metrics)
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

                with torch.no_grad():
                    model_forward_out = self.model(
                        iter_dict, is_training=False, is_validation=True, is_test=False
                    )
                loss = model_forward_out["loss"]
                self.validation_loss_meter.add_loss(loss, batch_size)
                self.validation_metric_calc.calc_metric(
                    iter_dict=iter_dict, model_forward_dict=model_forward_out
                )
            except StopIteration:
                self.validation_epoch_end(epoch_num)
                break

    def validation_epoch_end(self, epoch_num: int):

        self.msg_printer.divider("Validation @ Epoch {0}".format(epoch_num + 1))

        metrics = self.validation_metric_calc.report_metrics()

        average_loss = self.validation_loss_meter.get_average()
        print(metrics)

        self.msg_printer.text("Average Loss: {0}".format(average_loss))

        self.validation_logger.info(
            "Validation Loss @ Epoch {0} - {1}".format(epoch_num + 1, average_loss)
        )

        self.summaryWriter.add_scalars(
            "train_validation_loss",
            {"validation_loss": average_loss or np.inf},
            epoch_num + 1,
        )

        is_best: bool = None
        value_tracked: float = None

        if self.track_for_best == "loss":
            value_tracked = average_loss
            is_best = self.is_best_lower(average_loss)

        elif self.track_for_best == "macro-f1":
            macro_f1 = self.validation_metric_calc.get_metric()["macro_fscore"]
            value_tracked = macro_f1
            is_best = self.is_best_higher(current_best=macro_f1)
        elif self.track_for_best == "micro-f1":
            micro_f1 = self.validation_metric_calc.get_metric()["micro_fscore"]
            value_tracked = micro_f1
            is_best = self.is_best_higher(micro_f1)

        if self.lr_scheduler is not None:
            if self.lr_scheduler_is_plateau:
                self.lr_scheduler.step(value_tracked)
            else:
                self.lr_scheduler.step()

        if is_best:
            self.set_best_track_value(current_best=value_tracked)
            self.msg_printer.good(f"Found best model @ epoch {epoch_num + 1}")
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

                with torch.no_grad():
                    model_forward_out = self.model(
                        iter_dict, is_training=False, is_validation=False, is_test=True
                    )
                self.test_metric_calc.calc_metric(
                    iter_dict=iter_dict, model_forward_dict=model_forward_out
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
