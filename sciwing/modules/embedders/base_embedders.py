from abc import ABCMeta
from abc import abstractmethod
from sciwing.data.datasets_manager import DatasetsManager


class BaseEmbedder(metaclass=ABCMeta):
    def __init__(self, datasets_manager: DatasetsManager = None):
        self.dataset_manager = datasets_manager

    @abstractmethod
    def get_embedding_dimension(self):
        pass
