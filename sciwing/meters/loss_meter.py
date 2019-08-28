import numpy as np


class LossMeter:
    def __init__(self):
        self.losses = []
        self.batch_sizes = []

    def add_loss(self, avg_batch_loss: float, num_instances: int) -> None:
        """
        :param avg_batch_loss: type: float
        The average loss for the entire batch
        :param num_instances: type: int
        The total number of instances in the batch
        :return: None
        """
        self.losses.append(avg_batch_loss * num_instances)
        self.batch_sizes.append(num_instances)

    def get_average(self) -> float:
        if len(self.losses) == 0 or len(self.batch_sizes) == 0:
            average = None  # to indicate absent value
        else:
            average = sum(self.losses) / sum(self.batch_sizes)

        return average

    def reset(self):
        self.losses = []
        self.batch_sizes = []
