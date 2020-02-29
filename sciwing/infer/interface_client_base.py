from abc import ABCMeta
from abc import abstractmethod


class BaseInterfaceClient(metaclass=ABCMeta):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_dataset(self):
        pass

    @abstractmethod
    def build_infer(self):
        pass
