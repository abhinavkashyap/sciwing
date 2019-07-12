import torch
import torch.nn as nn
import wasabi


class LSTM2VecEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        embedding: torch.nn.Embedding,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        rnn_bias: bool = True,
    ):
        """

        :param emb_dim: type: int
        The embedding dimension of the embedding used in the first layer of the LSTM
        :param embedding: type: torch.nn.Embedding
        :param dropout_value: type: float
        :param hidden_dim: type: float
        :param bidirectional: type: bool
        Is the LSTM bi-directional
        :param combine_strategy: type: str
        This can be one of concat, sum, average
        This decides how different hidden states of different embeddings are combined
        :param rnn_bias: type: bool
        should RNN bias be used
        This is mainly for debugging purposes
        use sparingly
        """
        super(LSTM2VecEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = embedding
        self.dropout_value = dropout_value
        self.hidden_dimension = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.combine_strategy = combine_strategy
        self.allowed_combine_strategies = ["sum", "concat"]
        self.rnn_bias = rnn_bias
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
        x: torch.LongTensor,
        additional_embedding: torch.FloatTensor = None,
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Takes a sequence of tokens and converts into a vector representation
        There can be different functions to compute the vector representation
        \vec{s} = f([h_1... h_n]).
        In this class I have implemented only only \vec{s} = h_n the hidden representations
        from the last time step
        :param x: type: torch.LongTensor
        size: batch_size, num_tokens
        :param additional_embedding: type: torch.FloatTensor
        additional embedding is another embedding that can be concat
        with the embedding of x
        :param c0: type: torch.FloatTensor
        size: (num_layers * num_directions, batch_size, hidden_size)
        initial state
        :param h0: type: torch.FloatTensor
        size: (num_layers * num_directions, batch_size, hidden_size)
        Initial hidden state
        :return: type: Dict[str, Any]
        """
        assert x.ndimension() == 2, self.msg_printer.fail(
            f"LSTM2Vec expects a batch of tokens of "
            f"the shape batch_size * number_of_tokens."
            f"You passed a tensor of shape {x.shape}"
        )
        # batch_size * time steps * embedding dimension
        batch_size = x.size(0)
        embedded_tokens = self.embedding(x)
        embedded_tokens = self.emb_dropout(embedded_tokens)

        if additional_embedding is not None:
            embedded_tokens = torch.cat([embedded_tokens, additional_embedding], dim=2)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size, device=x.device)

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

    def get_initial_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dimension
        )
        h0 = h0.to(device)
        c0 = c0.to(device)
        return h0, c0
