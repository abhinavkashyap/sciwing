from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List
from sciwing.data.line import Line
from sciwing.data.label import Label
from sciwing.data.datasets_manager import DatasetsManager


class BaseMetric(metaclass=ABCMeta):
    def __init__(self, datasets_manager: DatasetsManager):
        pass

    @abstractmethod
    def calc_metric(
        self, lines: List[Line], labels: List[Label], model_forward_dict: Dict[str, Any]
    ) -> None:
        """ Calculates the metric using the lines and labels returned by any dataset
        and ``model_forward_dict`` of a model. This is usually called
        for a batch of inputs and a forward pass. The state of the different
        metrics should be retained by the metric across an epoch before
        ``reset`` method can be called and all the metric related data
        can be reset for a new epoch

        Parameters
        ----------
        lines : List[Line]
        labels: List[Label]
        model_forward_dict : Dict[str, Any]
        """
        pass

    @abstractmethod
    def get_metric(self) -> Dict[str, Any]:
        """Returns the value of different metrics being tracked

        Return anything that is being tracked by the metric.
        Return it as a dictionary that can be used by outside method
        for reporting purposes or repurposing it for the sake of reporting

        Returns
        -------
        Dict[str, Any]
            Metric/values being tracked by the metric
        """
        pass

    @abstractmethod
    def report_metrics(self, report_type: str = None) -> Any:
        """ A method to report the tracked metrics in a suitable form

        Parameters
        ----------
        report_type : str
            The type of report that will be returned by the method

        Returns
        -------
        Any
            This method can return any suitable format for reporting.
            If it is ought to be printed, return a suitable string.
            If the report needs to be saved to a file, go ahead.


        """

    @abstractmethod
    def reset(self):
        """ Should reset all the metrics/value being tracked by this metric
        This method is generally used at the end of a training/validation epoch
        to reset the values before starting another epoch
        """
        pass
