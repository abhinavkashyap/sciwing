from abc import ABCMeta, abstractmethod
import torch.nn as nn
import json
import torch
import wasabi
from typing import Dict, Any, Optional, Union, List


class BaseClassificationInference(metaclass=ABCMeta):
    """Abstract Base Class for Classification Inference.The BaseClassification Inference
    provides a skeleton for concrete classes that would want to perform inference for a
    text classification task.
    """

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
        model : nn.Module
            A pytorch module
        model_filepath : str
            The path where the parameters for the best models are stored. This is usually
            the ``best_model.pt`` while in an experiment directory
        dataset : Dataset
            Any dataset that conforms to the pytorch Dataset specification
        device : Optional[Union[str, torch.device]]
            This is either a string like ``cpu``, ``cuda:0`` or a torch.device object
        """
        self.model = model
        self.model_filepath = model_filepath
        self.dataset = dataset

        self.device = torch.device(device) if isinstance(device, str) else device
        self.msg_printer = wasabi.Printer()

    def load_model(self):
        """ Loads the best_model from the model_filepath.
        """
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
        """ Should Run inference on the test dataset

        This method should run the model through the test dataset.
        It should perform inference and collect the appropriate metrics
        and data that is necessary for further use

        Returns
        -------
        Dict[str, Any]
            Returns
        """
        pass

    @abstractmethod
    def model_forward_on_iter_dict(self, iter_dict: Dict[str, Any]):
        """ Perform the model forward pass  given an ``iter_dict``

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` returned by a dataset

        """
        pass

    @abstractmethod
    def metric_calc_on_iter_dict(
        self, iter_dict: Dict[str, Any], model_output_dict: Dict[str, Any]
    ):
        """ Calculate the metric given an ``iter_dict`` and an ``model_output_dict``
        that is obtained by a forward pass of the model

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` returned by a dataset

        model_output_dict : Dict[str, Any]
            ``model_output_dict`` : output dict that is returned by
            forwarding the ``iter_dict`` through the model

        """

    @abstractmethod
    def model_output_dict_to_prediction_indices_names(
        self, model_output_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        """ Given an ``model_output_dict``, it returns the predicted class indices and names

       Parameters
       ----------
       model_output_dict : Dict[str, Any]
           output dictionary from a model

       Returns
       -------
       (List[int], List[str])
           List of integers that represent the predicted class
           List of strings that represent the predicted class

       """

    @abstractmethod
    def iter_dict_to_sentences(self, iter_dict: Dict[str, Any]) -> List[str]:
        """ Returns human readable sentences given an ``iter_dict``

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` returned by a dataset

        Returns
        -------
        List[str]
            A list of human readable sentences

        """

    @abstractmethod
    def iter_dict_to_true_indices_names(
        self, iter_dict: Dict[str, Any]
    ) -> (List[int], List[str]):
        """ Given an ``iter_dict``, it returns the indices of the true classes
        and the corresponding classnames

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` returned by a dataset

        Returns
        -------
        (List[int], List[str])
            List of integers that represent the true class
            List of strings that represent the true class

        """

    @abstractmethod
    def report_metrics(self):
        """ Reports the metrics for returning the dataset
        """
        pass
