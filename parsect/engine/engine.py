from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


class Engine:
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset,
                 optimizer:optim,
                 batch_size: int,
                 save_dir: str):
        """
        This orchestrates the whole model training. The supervised machine learning
        that needs to be used
        :param model: type: nnn.Module
        Any pytorch model that takes in inputs
        :param train_dataset: type: torch.utils.data.Dataset
        training dataset, that can be iterated and follows the Dataset conventions of PyTorch
        :param validation_dataset: type: torch.utils.data.Dataset
        Validation dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param test_dataset: type: torch.utils.Dataset
        Test Dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param optimizer: torch.optim
        Any optimizer that belongs to the torch.optim module
        :param batch_size: type: int
        Batch size for the train, validation and test dataset
        :param save_dir: type: str
        The full location where the intermediate results are stored
        """

        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_dir = save_dir

        # get the data loader
        # TODO: For now we randomly sample the dataset to obtain instances
        self.train_loader = self.get_loader(self.train_dataset)
        self.validation_loader = self.get_loader(self.validation_dataset)
        self.test_loader =  self.get_loader(self.test_dataset)

    def get_loader(self, dataset: Dataset) -> DataLoader:
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size
        )
        return loader

