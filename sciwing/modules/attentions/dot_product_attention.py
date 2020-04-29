import torch.nn as nn
import torch
from sciwing.utils.class_nursery import ClassNursery


class DotProductAttention(nn.Module, ClassNursery):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        self.attn_softmax = nn.Softmax(dim=1)

    def forward(
        self, query_matrix: torch.Tensor, key_matrix: torch.Tensor
    ) -> torch.Tensor:
        """ Calculates the attention over the key

        Parameters
        ----------
        query_matrix: torch.Tensor
            Shape (batch_size, hidden_dimension)
        key_matrix: torch.Tensor
            Shape (batch_size, max_number_of_time_steps, hidden_dimension)

        Returns
        -------
        torch.Tensor
            The attention distribution over the keys
        """
        # (batch_size, hidden_dimension, 1)
        query_matrix_new_dimension = query_matrix.unsqueeze(-1)

        # (batch_size, max_number_time_steps, 1)
        attention = torch.bmm(key_matrix, query_matrix_new_dimension)

        # (batch_size, max_number_of_time_steps)
        attention = attention.squeeze(-1)

        # convert to probabilities
        attention = self.attn_softmax(attention)

        return attention
