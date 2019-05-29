import torch.nn as nn
from parsect.modules.elmo_embedder import ElmoEmbedder
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
import torch
from typing import List
import wasabi


class ElmoLSTMEncoder(nn.Module):
    def __init__(
        self,
        elmo_embedder: ElmoEmbedder,
    ):
        super(ElmoLSTMEncoder, self).__init__()
        self.elmo_embedder = elmo_embedder
        self.msg_printer = wasabi.Printer()

    def forward(self, x: torch.LongTensor, instances: List[List[str]]):
        assert x.size(0) == len(instances), self.msg_printer.fail(
            f"The batch size for tokens "
            f"and string instances should "
            f"be the same. You passed tokens of size {x.size()} and instances "
            f"have length {len(instances)}"
        )
        elmo_embeddings = self.elmo_embedder(instances)
