import torch
import torch.nn as nn
import wasabi
from typing import List
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line


class Lstm2SeqEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        embedder: nn.Module,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        num_layers: int = 1,
        combine_strategy: str = "concat",
        rnn_bias: bool = False,
        device: torch.device = torch.device("cpu"),
        add_projection_layer: bool = True,
        projection_activation: str = "Tanh",
    ):
        """Encodes a set of tokens to a set of hidden states.

        Parameters
        ----------
        embedder : nn.Module
            Any embedder can be used for this purpose
        dropout_value : float
            The dropout value for the embedding
        hidden_dim : int
            The hidden dimensions for the LSTM
        bidirectional : bool
            Whether the LSTM is bidirectional
        num_layers : int
            The number of layers of the LSTM
        combine_strategy : str
            The strategy to combine the different layers of the LSTM
            This can be one of
                sum
                    Sum the different layers of the embedding
                concat
                    Concat the layers of the embedding
        rnn_bias : bool
            Set this to false only for debugging purposes
        device : torch.device
        add_projection_layer: bool
            Adds a projection layer after the lstm over the hidden activation
        projection_activation: str
            Refer to torch.nn activations. Use any class name as a projection here
        """
        super(Lstm2SeqEncoder, self).__init__()
        self.embedder = embedder
        self.emb_dim = embedder.get_embedding_dimension()
        self.dropout_value = dropout_value
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.combine_strategy = combine_strategy
        self.rnn_bias = rnn_bias
        self.device = device
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.allowed_combine_strategies = ["sum", "concat"]
        self.msg_printer = wasabi.Printer()
        self.add_projection_layer = add_projection_layer
        self.projection_activation = projection_activation
        self.projection_activation_module = getattr(
            torch.nn, self.projection_activation
        )()

        assert (
            self.combine_strategy in self.allowed_combine_strategies
        ), self.msg_printer.fail(
            f"The combine strategies can be one of "
            f"{self.allowed_combine_strategies}. You passed "
            f"{self.combine_strategy}"
        )
        self.emb_dropout = nn.Dropout(p=self.dropout_value)
        self.output_dropout = nn.Dropout(p=self.dropout_value)
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            bias=self.rnn_bias,
            batch_first=True,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )
        if self.add_projection_layer:
            if self.combine_strategy == "concat" and bidirectional is True:
                output_hidden_dim = 2 * self.hidden_dim
            else:
                output_hidden_dim = self.hidden_dim
            self.projection_layer = nn.Linear(output_hidden_dim, hidden_dim)

    def forward(
        self,
        lines: List[Line],
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """

            Parameters
            ----------
            lines : List[Line]
                A list of lines
            c0 : torch.FloatTensor
                The initial state vector for the LSTM
            h0 : torch.FloatTensor
                The initial hidden state for the LSTM

            Returns
            -------
            torch.Tensor
                Returns the vector encoding of the set of instances
                [batch_size, seq_len, hidden_dim] if single direction
                [batch_size, seq_len, 2*hidden_dim] if bidirectional

        """

        embeddings = self.embedder(lines=lines)
        batch_size = len(lines)
        seq_length = embeddings.size(1)

        embeddings = self.emb_dropout(embeddings)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size)

        # output = batch_size, sequence_length, num_directions * hidden_size
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (hn, cn) = self.rnn(embeddings, (h0, c0))

        if self.bidirectional:
            output = output.view(batch_size, seq_length, self.num_directions, -1)
            forward_output = output[:, :, 0, :]  # batch_size, seq_length, hidden_dim
            backward_output = output[:, :, 1, :]  # batch_size, seq_length, hidden_dim
            if self.combine_strategy == "concat":
                encoding = torch.cat([forward_output, backward_output], dim=2)
            elif self.combine_strategy == "sum":
                encoding = torch.add(forward_output, backward_output)
            else:
                raise ValueError("The combine strategy should be one of concat or sum")
            hn = (
                hn.view(
                    self.num_layers, self.num_directions, batch_size, self.hidden_dim
                )
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(self.num_layers, batch_size, -1)
            )
            cn = (
                cn.view(
                    self.num_layers, self.num_directions, batch_size, self.hidden_dim
                )
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(self.num_layers, batch_size, -1)
            )
        else:
            encoding = output

        if self.add_projection_layer:
            encoding = self.projection_activation_module(
                self.projection_layer(encoding)
            )

        return encoding, (hn, cn)

    def get_initial_hidden(self, batch_size: int):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
