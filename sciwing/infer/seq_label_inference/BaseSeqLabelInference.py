from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
import wasabi
from sciwing.data.line import Line
from sciwing.data.seq_label import SeqLabel
import torch.nn as nn
from sciwing.data.datasets_manager import DatasetsManager


class BaseSeqLabelInference(metaclass=ABCMeta):
    """Abstract Base Class for Sequence Labeling Inference.The BaseSeqLabelInference Inference
    provides a skeleton for concrete classes that would want to perform inference for a
    text classification task.
    """

    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        datasets_manager: DatasetsManager,
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
        datasets_manager : DatasetsManager
            Any dataset that conforms to the pytorch Dataset specification
        device : Optional[Union[str, torch.device]]
            This is either a string like ``cpu``, ``cuda:0`` or a torch.device object
        """
        self.model = model
        self.model_filepath = model_filepath
        self.datasets_manager = datasets_manager
        self.device = device
        self.msg_printer = wasabi.Printer()

    @abstractmethod
    def run_inference(self):
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

    def load_model(self):
        """ Loads the best_model from the model_filepath.
        """
        model_chkpoint = torch.load(self.model_filepath, map_location=self.device)
        model_state_dict = model_chkpoint["model_state"]
        loss_value = model_chkpoint["loss"]
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.msg_printer.good(
            "Loaded Best Model with loss value {0}".format(loss_value)
        )

    @abstractmethod
    def model_forward_on_lines(self, lines: List[Line]):
        """ Perform the model forward pass  given an ``iter_dict``

        Parameters
        ----------
        lines : List[Line]
            ``iter_dict`` returned by a dataset

        """
        pass

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
    def get_true_label_indices_names(
        self, labels: List[SeqLabel]
    ) -> (Dict[str, List[int]], Dict[str, List[str]]):
        """ Given an list of labels, it returns the indices and the names of the label

        Parameters
        ----------
        labels : Dict[str, Any]
            ``iter_dict`` returned by a dataset

        Returns
        -------
        (Dict[str, List[int]], Dict[str, List[str]])
            A mapping between a label namespace and List of integers that represent the true class
            A mapping between a label namespace and a List of strings that represent the true class

        """

    @abstractmethod
    def report_metrics(self):
        """ Reports the metrics for returning the dataset
        """
        pass

    def run_test(self):
        pass

    def print_confusion_matrix(self):
        pass

    def get_misclassified_sentences(
        self, true_label_idx: int, pred_label_idx: int
    ) -> List[str]:
        pass

    def on_user_input(self, line: Union[Line, str]) -> Dict[str, List[str]]:
        pass

    def infer_batch(self, lines: Union[List[Line], List[str]]) -> Dict[str, List[str]]:
        pass
