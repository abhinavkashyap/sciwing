import torch.nn as nn
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.contextual_lines import LineWithContext
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from typing import List
import torch


class Lstm2SeqAttnContextEncoder(nn.Module, ClassNursery):
    def __init__(self, rnn2seqencoder: Lstm2SeqEncoder, attn_module: nn.Module):
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
        """
        super(Lstm2SeqAttnContextEncoder, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.attn_module = attn_module

    def forward(self, lines: List[LineWithContext]) -> torch.Tensor:

        # batch_size, number_of_time_steps, hidden_dimension
        encoding = self.rnn2seqencoder(lines=lines)
        num_timesteps = encoding.size(1)

        # batch_size, number_of_context_lines, hidden_dimension
        context_embedding = None

        # perform attention for every time step
        # the value is the context embedding
        # multiply the attention distribution over the context embedding
        # batch_size, number_of_time_steps, hidden_dimension

        attn_encoding = []
        for time_step in range(num_timesteps):
            query = encoding[time_step]

            # batch_size, number_of_context_lines
            attn = self.attn_module(query_matrix=query, key_matrix=context_embedding)

            attn_unsqueeze = attn.unsqueeze(1)

            # batch_size, 1, hidden_dimension
            values = torch.bmm(attn_unsqueeze, context_embedding)
            attn_encoding.append(values)

        # concatenate the representation
        # batch_size, number_of_time_steps, hidden_dimension
        final_encoding = torch.stack(attn_encoding, dim=1)
        final_encoding = torch.cat([encoding, final_encoding], dim=2)

        return final_encoding
