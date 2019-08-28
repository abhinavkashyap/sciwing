from allennlp.modules.elmo import Elmo, batch_to_ids
import sciwing.constants as constants
import torch.nn as nn
from typing import List
import wasabi
import torch

FILES = constants.FILES

ELMO_OPTIONS_FILE = FILES["ELMO_OPTIONS_FILE"]
ELMO_WEIGHTS_FILE = FILES["ELMO_WEIGHTS_FILE"]


class ElmoEmbedder(nn.Module):
    def __init__(
        self, dropout_value: float = 0.0, device: torch.device = torch.device("cpu")
    ):
        super(ElmoEmbedder, self).__init__()

        # Sometimes you need two different tensors that are
        # two different linear combination of representations
        # TODO: change this in-case you need 2 representations
        self.num_output_representations = 1
        self.dropout_value = dropout_value
        self.device = device
        self.msg_printer = wasabi.Printer()

        with self.msg_printer.loading("Loading Elmo Object"):
            self.elmo: nn.Module = Elmo(
                options_file=ELMO_OPTIONS_FILE,
                weight_file=ELMO_WEIGHTS_FILE,
                num_output_representations=self.num_output_representations,
                dropout=self.dropout_value,
            )

        self.msg_printer.good(f"Finished Loading ELMO object")

    def forward(self, x: List[List[str]]):
        character_ids = batch_to_ids(x)
        character_ids = character_ids.to(self.device)
        output_dict = self.elmo(character_ids)
        embeddings = output_dict["elmo_representations"][0]
        return embeddings
