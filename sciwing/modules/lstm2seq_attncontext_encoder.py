import torch.nn as nn
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.contextual_lines import LineWithContext
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from typing import List, Union
import torch


class Lstm2SeqAttnContextEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        attn_module: nn.Module,
        context_embedder: BaseEmbedder,
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """
        This module uses a lstm2seq encoder. The hidden dimensions can be
        enhanced by attention over some context for every line. Consider the context
        as something that is an additional information for the line. You can
        refer to LineWithContext for more information


        Parameters
        ----------
        rnn2seqencoder : Lstm2SeqEncoder
            Encodes the lines using lstm encoders to get contextual representations
        attn_module : nn.Module
            You can use attention modules from sciwing.modules.attention
        context_embedder: nn.Module
            You can use this embedder to embed context sentences
        device: Union[torch.device, str]
            The device on which this embedder is run
        """
        super(Lstm2SeqAttnContextEncoder, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.attn_module = attn_module
        self.context_embedder = context_embedder
        self.device = device

    def forward(self, lines: List[LineWithContext]) -> torch.Tensor:

        main_lines = []
        for line in lines:
            main_lines.append(line.line)

        # batch_size, number_of_time_steps, hidden_dimension
        encoding, _ = self.rnn2seqencoder(lines=main_lines)
        num_timesteps = encoding.size(1)

        # batch_size, max_num_context_lines, hidden_dimension
        max_num_context_lines = max([len(line.context_lines) for line in lines])
        context_embedding = []
        for line in lines:
            context_lines = line.context_lines
            num_context_lines = len(context_lines)
            embedding = self.context_embedder(lines=context_lines)
            # num_context_lines, embedding_dimension
            embedding = torch.mean(embedding, dim=1)
            emb_dim = embedding.size(1)

            # adding zeros for padding
            padding_length = max_num_context_lines - num_context_lines
            zeros = torch.randn(padding_length, emb_dim, device=self.device)

            embedding = torch.cat([embedding, zeros], dim=0)
            context_embedding.append(embedding)

        context_embedding = torch.stack(context_embedding)

        # perform attention for every time step
        # the value is the context embedding
        # multiply the attention distribution over the context embedding
        # batch_size, number_of_time_steps, hidden_dimension
        attn_encoding = []
        for time_step in range(num_timesteps):
            query = encoding[:, time_step, :]

            # batch_size, number_of_context_lines
            attn = self.attn_module(query_matrix=query, key_matrix=context_embedding)

            attn_unsqueeze = attn.unsqueeze(1)

            # batch_size, 1, hidden_dimension
            values = torch.bmm(attn_unsqueeze, context_embedding)

            # batch_size, hidden_dimension
            values = values.squeeze(1)
            attn_encoding.append(values)

        # concatenate the representation
        # batch_size, number_of_time_steps, hidden_dimension
        final_encoding = torch.stack(attn_encoding, dim=1)
        final_encoding = torch.cat([encoding, final_encoding], dim=2)

        return final_encoding
