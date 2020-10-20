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
        # vocab_size: int,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        num_layers: int = 1,
        combine_strategy: str = "concat",
        rnn_bias: bool = False,
        device: torch.device = torch.device("cpu")
    ):
        """Encodes a set of tokens to a set of hidden states.

        Parameters
        ----------
        embedder : nn.Module
            Any embedder can be used for this purpose
        vocab_size : int
            The size of the vocabulary from the datasetmanager
        dropout_value : float
            The dropout value for the embedding
        hidden_dim : int
            The hidden dimensions for the seq2seq encoder
        bidirectional: bool
            Whether the encoder is bidirectional. To decide the hidden state dimension.
        num_layers : int
            The number of layers of the LSTM
        combine_strategy : str
            The combine strategy of the encoder. If concat then the hidden state size of decoder is 2 * hidden_dim.
        rnn_bias : bool
            Set this to false only for debugging purposes
        device : torch.device
        """
        super(Lstm2SeqDecoder, self).__init__()
        self.embedder = embedder
        self.emb_dim = embedder.get_embedding_dimension()
        # self.vocab_size = vocab_size
        self.dropout_value = dropout_value
        self.hidden_dim = hidden_dim
        self.rnn_bias = rnn_bias
        self.device = device
        self.num_layers = num_layers
        self.msg_printer = wasabi.Printer()
        self.allowed_combine_strategies = ["sum", "concat"]

        self.emb_dropout = nn.Dropout(p=self.dropout_value)

        if bidirectional and combine_strategy == "concat":
            self.hidden_dim = self.hidden_dim * 2
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            bias=self.rnn_bias,
            batch_first=True,
            bidirectional=False,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
        )

        # self.output_size = self.vocab_size

        # self.output_layer = nn.Linear(self.hidden_dim, self.output_size)
        # self.softmax_layer = nn.LogSoftmax(dim=1)

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
                [batch_size, hidden_dim] if encoder is unidirectional
                [batch_size, 2*hidden_dim] if encoder is bidirectional
        """

        embeddings = self.embedder(lines=lines)
        batch_size = len(lines)
        seq_length = embeddings.size(1)

        embeddings = self.emb_dropout(embeddings)
        # embeddings = nn.functional.relu(embeddings)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size)

        # output = batch_size, sequence_length, num_directions * hidden_size
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (_, _) = self.rnn(embeddings, (h0, c0))

        # if self.bidirectional:
        #     output = output.view(batch_size, seq_length, self.num_directions, -1)
        #     forward_output = output[:, :, 0, :]  # batch_size, seq_length, hidden_dim
        #     backward_output = output[:, :, 1, :]  # batch_size, seq_length, hidden_dim
        #     if self.combine_strategy == "concat":
        #         encoding = torch.cat([forward_output, backward_output], dim=2)
        #     elif self.combine_strategy == "sum":
        #         encoding = torch.add(forward_output, backward_output)
        #     else:
        #         raise ValueError("The combine strategy should be one of concat or sum")
        # else:
        #     encoding = output

        # output = self.softmax_layer(self.output_layer(output))

        return output

    def get_initial_hidden(self, batch_size: int):
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim
        )
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
