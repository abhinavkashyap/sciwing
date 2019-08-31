import numpy as np


class LossMeter:
    def __init__(self):
        self.losses = []
        self.batch_sizes = []

    def add_loss(self, avg_batch_loss: float, num_instances: int) -> None:
        """ Adds the average batch loss and the num of instances in that batch to that loss

        Parameters
        ----------
        avg_batch_loss : float
            Average batch loss
        num_instances : int
            Number of instances from the batch

        """
        self.losses.append(avg_batch_loss * num_instances)
        self.batch_sizes.append(num_instances)

    def get_average(self) -> float:
        """ Returns the average loss over all the batches at this point in time

        Returns
        -------
        float:
            Average loss

        """
        if len(self.losses) == 0 or len(self.batch_sizes) == 0:
            average = None  # to indicate absent value
        else:
            average = sum(self.losses) / sum(self.batch_sizes)

        return average

    def reset(self):
        """ Resets all the losses and batch sizes that are accumulated
        """
        self.losses = []
        self.batch_sizes = []
