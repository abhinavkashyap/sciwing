import torch
import torch.nn as nn
from typing import Dict, Any
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.utils.class_nursery import ClassNursery
from deprecated import deprecated


@deprecated(reason="We have a new char encoder that is easier to use", version=0.1)
class CharLSTMEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        char_embedder: nn.Module,
        char_emb_dim: int,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        device: torch.device = torch.device("cpu"),
    ):
        """ Encodes character tokens using lstms

        Parameters
        ----------
        char_embedder : nn.Module
            An embedder that embeds character tokens
        char_emb_dim : int
            The embedding of characters
        hidden_dim : int
            Hidden dimension of the LSTM
        bidirectional : bool
            Should the LSTM be bi-directional
        combine_strategy : str
            Combine strategy for the lstm hidden dimensions
        device : torch.device("cpu)
            The device on which the lstm will run
        """
        super(CharLSTMEncoder, self).__init__()
        self.char_embedder = char_embedder
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.combine_strategy = combine_strategy
        self.device = device
        self.seq2vecencoder = LSTM2VecEncoder(
            embedder=self.char_embedder,
            emb_dim=char_emb_dim,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            combine_strategy=combine_strategy,
            rnn_bias=True,
            device=device,
        )

    def forward(self, iter_dict: Dict[str, Any]):
        """

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            expects char_tokens to be present in the ``iter_dict``
            from any dataset
        Returns
        -------
        torch.Tensor:
            ``[batch_size, num_time_steps, hidden_dim]``
            The hidden dimension is the hidden dimension of the LSTM
            if it is bidirectional and concat then ``hidden_dim``
            will be `2 * self.hidden_dim`

        """
        char_tokens = iter_dict["char_tokens"]
        assert (
            char_tokens.dim() == 3
        ), f"char_tokens passed to CharLSTMEncoder should be of 3 dimensions"
        batch_size, num_time_steps, max_num_chars = char_tokens.size()
        char_tokens = char_tokens.view(batch_size * num_time_steps, max_num_chars)
        embedding = self.seq2vecencoder(iter_dict={"tokens": char_tokens})
        embedding = embedding.view(batch_size, num_time_steps, -1)
        return embedding
