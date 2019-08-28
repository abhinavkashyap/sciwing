from abc import ABCMeta, abstractmethod
import torch.nn as nn
import json
import torch
import wasabi
from typing import Dict, Any, Optional, Union, List


class BaseClassificationInference(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        dataset,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        """

        Parameters
        ----------
        model
        model_filepath
        dataset
        device
        """
        self.model = model
        self.model_filepath = model_filepath
        self.dataset = dataset

        self.device = torch.device(device) if isinstance(device, str) else device
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

    @abstractmethod
    def model_forward_on_iter_dict(self, iter_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def metric_calc_on_iter_dict(
        self, iter_dict: Dict[str, Any], model_output_dict: Dict[str, Any]
    ):
        pass

    @abstractmethod
    def model_output_dict_to_prediction_indices_names(
        self, model_output_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        pass

    @abstractmethod
    def iter_dict_to_sentences(self, iter_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def iter_dict_to_true_indices_names(self, iter_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def print_metrics(self):
        pass
