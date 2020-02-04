import torch.nn as nn
from sciwing.modules.embedders.elmo_embedder import ElmoEmbedder
import torch
from typing import List
import wasabi
from deprecated import deprecated


@deprecated(
    reason="ElmoEmbedder will be deprecated "
    "Please use concat embedder and lstm 2 vec module to achieve the same thing. "
    "This will be removed in version 1",
    version=0.1,
)
class ElmoLSTMEncoder(nn.Module):
    def __init__(
        self,
        elmo_emb_dim: int,
        elmo_embedder: ElmoEmbedder,
        emb_dim: int,
        embedding: torch.nn.Embedding,
        dropout_value: float,
        hidden_dim: int,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        rnn_bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super(ElmoLSTMEncoder, self).__init__()
        self.elmo_emb_dim = elmo_emb_dim
        self.elmo_embedder = elmo_embedder
        self.emb_dim = emb_dim
        self.embedding = embedding
        self.dropout_value = dropout_value
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.combine_strategy = combine_strategy
        self.rnn_bias = rnn_bias
        self.allowed_combine_strategies = ["sum", "concat"]
        self.num_layers = 1
        self.num_directions = 2 if self.bidirectional else 1
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
            input_size=self.emb_dim + self.elmo_emb_dim,
            hidden_size=self.hidden_dim,
            bias=self.rnn_bias,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def forward(
        self,
        x: torch.LongTensor,
        instances: List[List[str]],
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ):
        assert x.size(0) == len(instances), self.msg_printer.fail(
            f"The batch size for tokens "
            f"and string instances should "
            f"be the same. You passed tokens of size {x.size()} and instances "
            f"have length {len(instances)}"
        )
        batch_size = x.size(0)
        elmo_embeddings = self.elmo_embedder(instances)
        token_embeddings = self.embedding(x)

        # concat both of them together
        embeddings = torch.cat([elmo_embeddings, token_embeddings], dim=2)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size)

        # output = batch_size, sequence_length, num_layers * num_directions
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (h_n, c_n) = self.rnn(embeddings, (h0, c0))

        encoding = None
        if self.bidirectional:
            forward_hidden = h_n[0, :, :]
            backward_hidden = h_n[1, :, :]
            if self.combine_strategy == "concat":
                encoding = torch.cat([forward_hidden, backward_hidden], dim=1)
            elif self.combine_strategy == "sum":
                encoding = torch.add(forward_hidden, backward_hidden)
        else:
            encoding = h_n[0, :, :]

        # N * hidden_dim
        return encoding

    def get_initial_hidden(self, batch_size):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
