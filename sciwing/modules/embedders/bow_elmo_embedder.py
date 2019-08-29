import torch
from allennlp.commands.elmo import ElmoEmbedder
import wasabi
from typing import List, Iterable, Dict, Any
import torch.nn as nn
from sciwing.utils.class_nursery import ClassNursery


class BowElmoEmbedder(nn.Module, ClassNursery):
    def __init__(
        self,
        emb_dim: int = 1024,
        dropout_value: float = 0.0,
        layer_aggregation: str = "sum",
        cuda_device_id: int = -1,
    ):
        super(BowElmoEmbedder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.layer_aggregation_type = layer_aggregation
        self.allowed_layer_aggregation_types = ["sum", "average", "last", "first"]
        self.cuda_device_id = cuda_device_id
        self.msg_printer = wasabi.Printer()

        assert (
            self.layer_aggregation_type in self.allowed_layer_aggregation_types
        ), self.msg_printer.fail(
            f"For bag of words elmo encoder, the allowable aggregation "
            f"types are {self.allowed_layer_aggregation_types}. You passed {self.layer_aggregation_type}"
        )

        # load the elmo embedders
        with self.msg_printer.loading("Creating Elmo object"):
            self.elmo = ElmoEmbedder(cuda_device=self.cuda_device_id)
        self.msg_printer.good("Finished Loading Elmo object")

    def forward(self, iter_dict: Dict[str, Any]) -> torch.Tensor:
        # [np.array] - A generator of embeddings
        # each array in the list is of the shape (3, #words_in_sentence, 1024)
        x = iter_dict["instance"]
        x = x if isinstance(x, list) else [x]
        x = [instance.split() for instance in x]

        print(f"x {x}")

        embedded = list(self.elmo.embed_sentences(x))

        # bs, 3, #words_in_sentence, 1024
        embedded = torch.FloatTensor(embedded)

        embedding_ = None
        # aggregate of word embeddings
        if self.layer_aggregation_type == "sum":
            # bs, #words_in_sentence, 1024
            embedding_ = torch.sum(embedded, dim=1)

        elif self.layer_aggregation_type == "average":
            # mean across all layers
            embedding_ = torch.mean(embedded, dim=1)

        elif self.layer_aggregation_type == "last":
            # bs, max_len, 1024
            embedding_ = embedded[:, -1, :, :]

        elif self.layer_aggregation_type == "first":
            # bs, max_len, 1024
            embedding_ = embedded[:, 0, :, :]

        return embedding_

    def __call__(self, iter_dict: Dict[str, Any]):
        return self.forward(iter_dict)
