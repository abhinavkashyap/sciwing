import torch
import torch.nn as nn
from typing import Dict, Any
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder


class CharLSTMEncoder(nn.Module):
    def __init__(
        self,
        char_embedder: nn.Module,
        char_emb_dim: int,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        device: torch.device = torch.device("cpu"),
    ):
        super(CharLSTMEncoder, self).__init__()
        self.char_embedder = char_embedder
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.combine_strategy = combine_strategy
        self.device = device
        self.seq2vecencoder = LSTM2VecEncoder(
            embedder=self.char_embedder,
            emb_dim=char_emb_dim,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            combine_strategy=combine_strategy,
            rnn_bias=True,
            device=device,
        )

    def forward(self, iter_dict: Dict[str, Any]):
        char_tokens = iter_dict["char_tokens"]
        assert (
            char_tokens.dim() == 3
        ), f"char_tokens passed to CharLSTMEncoder should be of 3 dimensions"
        batch_size, num_time_steps, max_num_chars = char_tokens.size()
        char_tokens = char_tokens.view(batch_size * num_time_steps, max_num_chars)
        embedding = self.seq2vecencoder(iter_dict={"tokens": char_tokens})
        embedding = embedding.view(batch_size, num_time_steps, -1)
        return embedding
