from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
import wasabi


class BaseSeqLabelInference(metaclass=ABCMeta):
    def __init__(
        self, model, model_filepath, dataset, device: Optional[Union[str, torch.device]]
    ):
        self.model = model
        self.model_filepath = model_filepath
        self.dataset = dataset
        self.device = device
        self.msg_printer = wasabi.Printer()

    @abstractmethod
    def run_inference(self):
        pass

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

    @abstractmethod
    def run_test(self):
        pass
