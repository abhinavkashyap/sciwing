import torch
import torch.nn as nn
import wasabi
from typing import List
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.data.datasets_manager import DatasetsManager
import random
from sciwing.vocab.vocab import Vocab


class Lstm2SeqDecoder(nn.Module, ClassNursery):
    def __init__(
        self,
        embedder: nn.Module,
        vocab: Vocab,
        max_length: int,
        attn_module: nn.Module = None,
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
        self.vocab = vocab
        self.max_length = max_length
        self.attn_module = attn_module
        self.emb_dim = embedder.get_embedding_dimension()
        self.dropout_value = dropout_value
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
        self.vocab_size = self.vocab.get_vocab_len()
        self.start_token = self.vocab.start_token
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

        if self.attn_module:
            self.projection_layer = nn.Linear(self.hidden_dim * 2, self.vocab_size)
        else:
            self.projection_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward_step(
        self,
        lines: List[Line],
        c0: torch.FloatTensor,
        h0: torch.FloatTensor,
        encoder_outputs: torch.FloatTensor = None,
    ) -> (torch.Tensor, torch.FloatTensor, torch.FloatTensor):
        """

        Parameters
        ----------
        lines : list of Line objects
            Batched tokenized source sentence of shape [batch size].

        h0, c0 : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]

        Returns
        -------
        prediction : 2d torch.LongTensor
            For each token in the batch, the predicted target vobulary.
            Shape [batch size, output dim]

        hn, cn : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]
        """
        # [batch_size, seq_length, emb dim], the 1 serves as sent len
        embeddings = self.embedder(lines=lines)
        embeddings = self.emb_dropout(embeddings)

        outputs, (hn, cn) = self.rnn(embeddings, (h0, c0))

        if self.attn_module:
            attn_encoding_list = []
            for time_step in range(outputs.size(1)):
                query = outputs[:, time_step, :]

                # batch_size, number_of_context_lines
                attn = self.attn_module(query_matrix=query, key_matrix=encoder_outputs)

                attn_unsqueeze = attn.unsqueeze(1)

                # batch_size, 1, hidden_dimension
                values = torch.bmm(attn_unsqueeze, encoder_outputs)
                #
                # # batch_size, 1, 2 * hidden_dimension
                # outputs = torch.cat((values, outputs), -1)

                values = values.squeeze(1)
                attn_encoding_list.append(values)

            # concatenate the representation
            # batch_size, number_of_time_steps, hidden_dimension
            attn_encoding = torch.stack(attn_encoding_list, dim=1)
            outputs = torch.cat([outputs, attn_encoding], dim=2)

        prediction = nn.functional.softmax(self.projection_layer(outputs))

        return prediction, hn, cn

    def forward(
        self,
        lines: List[Line],
        c0: torch.FloatTensor,
        h0: torch.FloatTensor,
        encoder_outputs: torch.FloatTensor = None,
        teacher_forcing_ratio: float = 0,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        lines : list of Line objects
            Batched tokenized source sentence of shape [batch size].

        h0, c0 : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]

        Returns
        -------
        prediction : 2d torch.LongTensor
            For each token in the batch, the predicted target vobulary.
            Shape [batch size, output dim]

        hn, cn : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]
        """
        use_teacher_forcing = (
            True if (random.random() < teacher_forcing_ratio) else False
        )
        if use_teacher_forcing:
            max_length = max(len(line.tokens["tokens"]) for line in lines)
        else:
            max_length = self.max_length
        batch_size = len(lines)

        # tensor to store decoder's output
        outputs = torch.zeros(max_length, batch_size, self.vocab_size).to(self.device)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        if use_teacher_forcing:
            prediction, _, _ = self.forward_step(
                lines=lines, h0=h0, c0=c0, encoder_outputs=encoder_outputs
            )
            outputs[1:] = prediction.permute(1, 0, 2)[:-1]
        else:
            lines = [self._generate_lines_with_start_token()] * batch_size
            for i in range(1, max_length):
                prediction, hn, cn = self.forward_step(
                    lines=lines, h0=h0, c0=c0, encoder_outputs=encoder_outputs
                )
                prediction = prediction.squeeze(1)
                outputs[i] = prediction
                line_token_indexes = prediction.argmax(1)
                line_tokens = [
                    self.vocab.idx2token[line_token_index]
                    for line_token_index in line_token_indexes.cpu().numpy()
                ]
                lines = []
                for token in line_tokens:
                    line = Line("")
                    line.add_token(token, "tokens")
                    lines.append(line)
                h0, c0 = hn, cn
        outputs = outputs.permute(1, 0, 2)
        return outputs

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

    def _generate_lines_with_start_token(self):
        line = Line("")
        line.add_token(self.start_token, "tokens")
        return line
