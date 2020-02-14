import torch.nn as nn
from typing import Dict, List
from torchcrf import CRF
import torch
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.data.seq_label import SeqLabel
from sciwing.data.line import Line
from collections import defaultdict


class ParscitTagger(nn.Module):
    """PyTorch module for Neural Parscit"""

    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        encoding_dim: int,
        datasets_manager: DatasetsManager,
        device: torch.device = torch.device("cpu"),
    ):
        """

        Parameters
        ----------
        rnn2seqencoder : Lstm2SeqEncoder
            Lstm2SeqEncoder that encodes a set of instances to a sequence of hidden states
        encoding_dim : int
            Hidden dimension of the lstm2seq encoder
        """
        super(ParscitTagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.encoding_dim = encoding_dim
        self.datasets_manager = datasets_manager
        self.label_namespaces = datasets_manager.label_namespaces
        self.device = device
        self.crfs = {}
        self.linear_clfs = {}
        for namespace in self.label_namespaces:
            num_labels = self.datasets_manager.num_labels[namespace]
            crf = CRF(num_tags=num_labels, batch_first=True)
            clf = nn.Linear(self.encoding_dim, num_labels)
            self.crfs[namespace] = crf
            self.linear_clfs[namespace] = clf

    def forward(
        self,
        lines: List[Line],
        labels: List[SeqLabel],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
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
        encoding = self.rnn2seqencoder(lines=lines)
        max_time_steps = encoding.size(1)

        # batch size, time steps, num_classes
        output_dict = {}
        for namespace in self.label_namespaces:
            namespace_logits = self.linear_clfs[namespace](encoding)
            output_dict[f"logits_{namespace}"] = namespace_logits
            output_dict[f"predicted_tags_{namespace}"] = self.crfs[namespace].decode(
                namespace_logits
            )

        if is_training or is_validation:
            labels_indices = defaultdict(list)
            for label in labels:
                for namespace in self.label_namespaces:
                    numericalizer = self.datasets_manager.namespace_to_numericalizer[
                        namespace
                    ]
                    label_ = label.tokens[namespace]
                    label_ = [tok.text for tok in label_]
                    label_instances = numericalizer.numericalize_instance(
                        instance=label_
                    )
                    label_instances = numericalizer.pad_instance(
                        numericalized_text=label_instances,
                        max_length=max_time_steps,
                        add_start_end_token=False,
                    )
                    label_instances = torch.LongTensor(label_instances)
                    label_instances = label_instances.to(self.device)
                    labels_indices[namespace].append(label_instances)

            losses = []
            for namespace in self.label_namespaces:
                labels_tensor = labels_indices[namespace]
                labels_tensor = torch.stack(labels_tensor)
                loss_ = -self.crfs[namespace](
                    output_dict[f"logits_{namespace}"], labels_tensor
                )
                losses.append(loss_)

            loss = sum(losses)
            output_dict["loss"] = loss

        return output_dict
