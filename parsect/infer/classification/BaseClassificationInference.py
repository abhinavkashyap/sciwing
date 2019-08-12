from abc import ABCMeta, abstractmethod
import torch.nn as nn
import json
import torch
import wasabi
from typing import Dict, Any


class BaseClassificationInference(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        hyperparam_config_filepath: str,
        dataset,
    ):
        """
               :param model: type: torch.nn.Module
               Pass the model on which inference should be run
               :param model_filepath: type: str
               The model filepath is the chkpoint file where the model state is stored
               :param hyperparam_config_filepath: type: str
               The path where all hyper-parameters necessary for restoring the model
               is necessary
        """
        self.model = model
        self.model_filepath = model_filepath
        self.hyperparam_config_filename = hyperparam_config_filepath
        self.test_dataset = dataset

        with open(self.hyperparam_config_filename, "r") as fp:
            config = json.load(fp)

        self.device = torch.device(config.get("DEVICE", "cpu"))
        self.msg_printer = wasabi.Printer()

    def load_model(self):

        model_chkpoint = torch.load(self.model_filepath)
        model_state_dict = model_chkpoint["model_state"]
        loss_value = model_chkpoint["loss"]
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.msg_printer.good(
            "Loaded Best Model with loss value {0}".format(loss_value)
        )

    @abstractmethod
    def run_inference(self) -> Dict[str, Any]:
        pass
