from allennlp.modules.elmo import Elmo, batch_to_ids
import sciwing.constants as constants
import torch.nn as nn
from typing import List
import wasabi
import torch
from sciwing.data.line import Line
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager

FILES = constants.FILES

ELMO_OPTIONS_FILE = FILES["ELMO_OPTIONS_FILE"]
ELMO_WEIGHTS_FILE = FILES["ELMO_WEIGHTS_FILE"]


class ElmoEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        dropout_value: float = 0.5,
        datasets_manager: DatasetsManager = None,
        word_tokens_namespace: str = "tokens",
        device: torch.device = torch.device("cpu"),
        fine_tune: bool = False,
    ):
        super(ElmoEmbedder, self).__init__()

        # Sometimes you need two different tensors that are
        # two different linear combination of representations
        # TODO: change this in-case you need 2 representations
        self.num_output_representations = 1
        self.dropout_value = dropout_value
        self.datasets_manager = datasets_manager
        self.device = torch.device(device) if isinstance(device, str) else device
        self.msg_printer = wasabi.Printer()
        self.word_tokens_namespace = word_tokens_namespace
        self.fine_tune = fine_tune
        self.embedder_name = "ElmoEmbedder"

        with self.msg_printer.loading("Loading Elmo Object"):
            self.elmo: nn.Module = Elmo(
                options_file=ELMO_OPTIONS_FILE,
                weight_file=ELMO_WEIGHTS_FILE,
                num_output_representations=self.num_output_representations,
                dropout=self.dropout_value,
                requires_grad=fine_tune,
            )

        self.msg_printer.good(f"Finished Loading ELMO object")

    def forward(self, lines: List[Line]):
        texts = []
        for line in lines:
            line_tokens = line.tokens[self.word_tokens_namespace]
            line_tokens = list(map(lambda tok: tok.text, line_tokens))
            texts.append(line_tokens)

        character_ids = batch_to_ids(texts)
        character_ids = character_ids.to(self.device)
        output_dict = self.elmo(character_ids)
        # batch_size, max_seq_length * 1024
        embeddings = output_dict["elmo_representations"][0]
        return embeddings

    def get_embedding_dimension(self):
        return 1024
