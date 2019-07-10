from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Union


class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def calc_metric(
        self, iter_dict: Dict[str, Any], model_forward_dict: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    def get_metric(self) -> Dict[str, Union[Dict[str, float], float]]:
        pass

    @abstractmethod
    def report_metrics(self, report_type="wasabi") -> None:
        pass

    @abstractmethod
    def reset(self):
        pass
