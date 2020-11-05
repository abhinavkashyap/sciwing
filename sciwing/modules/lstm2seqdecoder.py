import torch
import torch.nn as nn
import wasabi
from typing import List
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.data.datasets_manager import DatasetsManager


class Lstm2SeqDecoder(nn.Module, ClassNursery):
    def __init__(
        self,
        embedder: nn.Module,
        vocab_size: int,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        num_layers: int = 1,
        combine_strategy: str = "concat",
        rnn_bias: bool = False,
        device: torch.device = torch.device("cpu"),
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
        projection_activation: str
            Refer to torch.nn activations.
        output_size : int
            Vocabulary size
        """
        super(Lstm2SeqDecoder, self).__init__()
        self.embedder = embedder
        self.vocab_size = vocab_size
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
        self.projection_activation = projection_activation
        self.projection_activation_module = getattr(
            torch.nn, self.projection_activation
        )()
        if self.combine_strategy == "concat" and bidirectional is True:
            self.hidden_dim = 2 * hidden_dim
        else:
            self.hidden_dim = hidden_dim

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
            bidirectional=False,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )

        self.projection_layer = nn.Linear(self.hidden_dim, vocab_size)

    def forward(
        self,
        lines: List[Line],
        c0: torch.FloatTensor,
        h0: torch.FloatTensor,
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

        # output = batch_size, sequence_length, num_directions * hidden_size
        # h_n = num_layers, batch_size, num_directions * hidden_dimension
        # c_n = num_layers, batch_size, num_directions * hidden_dimension
        output, (_, _) = self.rnn(embeddings, (h0, c0))

        predictions = self.projection_activation_module(self.projection_layer(output))

        return predictions

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

