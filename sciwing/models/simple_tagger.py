import torch.nn as nn
from typing import Dict, List
import torch
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.data.datasets_manager import DatasetsManager
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from sciwing.data.seq_label import SeqLabel
from sciwing.data.line import Line
from sciwing.utils.class_nursery import ClassNursery


class SimpleTagger(nn.Module, ClassNursery):
    """PyTorch module for Neural Parscit"""

    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        encoding_dim: int,
        datasets_manager: DatasetsManager,
        device: torch.device = torch.device("cpu"),
        label_namespace: str = "seq_label",
    ):
        """

        Parameters
        ----------
        rnn2seqencoder : Lstm2SeqEncoder
            Lstm2SeqEncoder that encodes a set of instances to a sequence of hidden states
        encoding_dim : int
            Hidden dimension of the lstm2seq encoder
        """
        super(SimpleTagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.encoding_dim = encoding_dim
        self.datasets_manager = datasets_manager

        self.label_namespace = datasets_manager.label_namespaces[0]
        self.device = device
        self.num_labels = self.datasets_manager.num_labels[self.label_namespace]

        self.linear_proj = nn.Linear(self.encoding_dim, self.num_labels)
        self._loss = CrossEntropyLoss()

    def forward(
        self,
        lines: List[Line],
        labels: List[SeqLabel] = None,
        is_training: bool = False,
        is_validation: bool = False,
        is_test: bool = False,
    ):
        """
        Parameters
        ----------
        lines : List[lines]
            A list of lines
        labels: List[SeqLabel]
            A list of sequence labels
        is_training : bool
            running forward on training dataset?
        is_validation : bool
            running forward on training dataset ?
        is_test : bool
            running forward on test dataset?


        Returns
        -------
        Dict[str, Any]
            logits: torch.FloatTensor
                Un-normalized probabilities over all the classes
                of the shape ``[batch_size, num_classes]``
            predicted_tags: List[List[int]]
                Set of predicted tags for the batch
            loss: float
                Loss value if this is a training forward pass
                or validation loss. There will be no loss
                if this is the test dataset
        """

        # batch size, max_num_word_tokens, hidden_dim
        encoding, _ = self.rnn2seqencoder(lines=lines)
        max_time_steps = encoding.size(1)
        batch_size = encoding.size(0)

        output_dict = {}

        # batch size, time steps, num_classes
        namespace_logits = self.linear_proj(encoding)
        normalized_probs = softmax(namespace_logits, dim=2)

        batch_size, time_steps, _ = namespace_logits.size()
        output_dict[f"logits_{self.label_namespace}"] = namespace_logits
        output_dict["normalized_probs"] = normalized_probs
        predicted_tags = torch.topk(namespace_logits, k=1, dim=2)

        # gets the max element indices and flattens it to get List[List[int]]
        predicted_tags = predicted_tags.indices.flatten(start_dim=1).tolist()
        output_dict[f"predicted_tags_{self.label_namespace}"] = predicted_tags

        labels_indices = []
        if is_training or is_validation:
            for label in labels:
                numericalizer = self.datasets_manager.namespace_to_numericalizer[
                    self.label_namespace
                ]
                label_ = label.tokens[self.label_namespace]
                label_ = [tok.text for tok in label_]
                label_instances = numericalizer.numericalize_instance(instance=label_)
                label_instances = numericalizer.pad_instance(
                    numericalized_text=label_instances,
                    max_length=max_time_steps,
                    add_start_end_token=False,
                )
                label_instances = torch.tensor(
                    label_instances, dtype=torch.long, device=self.device
                )
                labels_indices.append(label_instances)

            # batch_size, num_steps
            labels_tensor = torch.stack(labels_indices)
            loss = self._loss(
                input=normalized_probs.view(batch_size * max_time_steps, -1),
                target=labels_tensor.view(-1),
            )

            output_dict["loss"] = loss

        return output_dict
