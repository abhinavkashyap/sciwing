import torch
import torch.nn as nn
import wasabi
from typing import Union, List
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery


class LSTM2VecEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        embedder,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        rnn_bias: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """LSTM2Vec encoder that encodes a series of tokens to a single vector representation

        Parameters
        ----------
        embedder : nn.Module
            Any embedder can be passed
        dropout_value : float
            The dropout value for input embeddings
        hidden_dim : int
            The hidden dimension for the LSTM
        bidirectional : bool
            Whether the LSTM is bidirectional or no
        combine_strategy : str
            Strategy to combine the vectors from two different directions
        rnn_bias : str
            Whether to use the bias layer in RNN. Should be set to false only for debugging purposes
        device : Union[str, torch.device]
            The device on which the model is run
        """
        super(LSTM2VecEncoder, self).__init__()
        self.embedder = embedder
        self.emb_dim = embedder.get_embedding_dimension()
        self.dropout_value = dropout_value
        self.hidden_dimension = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.combine_strategy = combine_strategy
        self.allowed_combine_strategies = ["sum", "concat"]
        self.rnn_bias = rnn_bias
        self.device = torch.device(device) if isinstance(device, str) else device
        self.msg_printer = wasabi.Printer()

        assert (
            self.combine_strategy in self.allowed_combine_strategies
        ), self.msg_printer.fail(
            f"The combine strategies can be one of "
            f"{self.allowed_combine_strategies}. You passed "
            f"{self.combine_strategy}"
        )

        self.emb_dropout = nn.Dropout(p=self.dropout_value)
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dimension,
            bias=self.rnn_bias,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def forward(
        self,
        lines: List[Line],
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """

        Parameters
        ----------
        lines: List[Line]
            A list of lines to be encoder
        c0 : torch.FloatTensor
            The initial state vector for the LSTM
        h0 : torch.FloatTensor
            The initial hidden state for the LSTM

        Returns
        -------
        torch.Tensor
            Returns the vector encoding of the set of instances
            [batch_size, hidden_dim] if single direction
            [batch_size, 2*hidden_dim] if bidirectional
        """

        batch_size = len(lines)
        embedded_tokens = self.embedder(lines)
        embedded_tokens = self.emb_dropout(embedded_tokens)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size)

        # output = batch_size, sequence_length, num_layers * num_directions
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (h_n, c_n) = self.rnn(embedded_tokens, (h0, c0))

        # for a discourse on bi-directional RNNs and LSTMS
        # and what h_n and c_n contain refer to
        # https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        # h_n forward contain hidden states after processing x_1 -> x_2... -> x_n
        # h_n backward contains hidden states after processing x_n -> x_{n-1} -> ... x_1
        if self.bidirectional:
            forward_hidden = h_n[0, :, :]
            backward_hidden = h_n[1, :, :]
            if self.combine_strategy == "concat":
                encoding = torch.cat([forward_hidden, backward_hidden], dim=1)
            elif self.combine_strategy == "sum":
                encoding = torch.add(forward_hidden, backward_hidden)
            else:
                raise ValueError(f"The combine strategy should be one of concat or sum")
        else:
            encoding = h_n[0, :, :]

        return encoding

    def get_initial_hidden(self, batch_size: int):
        """ Gets the initial hidden states of the LSTM2Vec encoder

        Parameters
        ----------
        batch_size : int
            The batch size of the current forward pass

        Returns
        -------
        torch.Tensor, torch.Tensor
        """
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
