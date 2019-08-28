import torch
import torch.nn as nn
import wasabi
from typing import Dict, Any
from sciwing.utils.class_nursery import ClassNursery


class LSTM2VecEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        emb_dim: int,
        embedder,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        rnn_bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super(LSTM2VecEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.embedder = embedder
        self.dropout_value = dropout_value
        self.hidden_dimension = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.combine_strategy = combine_strategy
        self.allowed_combine_strategies = ["sum", "concat"]
        self.rnn_bias = rnn_bias
        self.device = device
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
        iter_dict: Dict[str, Any],
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        batch_size = iter_dict["tokens"].size(0)
        embedded_tokens = self.embedder(iter_dict)
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
            encoding = h_n[0, :, :]

        return encoding

    def get_initial_hidden(self, batch_size: int):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
