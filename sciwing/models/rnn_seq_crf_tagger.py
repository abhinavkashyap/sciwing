import torch.nn as nn
from typing import Dict, List, Tuple
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from allennlp.modules.conditional_random_field import allowed_transitions
import torch
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.data.seq_label import SeqLabel
from sciwing.data.line import Line
from collections import defaultdict
from sciwing.utils.class_nursery import ClassNursery


class RnnSeqCrfTagger(nn.Module, ClassNursery):
    """PyTorch module for Neural Parscit"""

    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        encoding_dim: int,
        datasets_manager: DatasetsManager,
        device: torch.device = torch.device("cpu"),
        namespace_to_constraints: Dict[str, List[Tuple[int, int]]] = None,
        tagging_type=None,
        include_start_end_trainsitions: bool = True,
    ):
        """

        Parameters
        ----------
        rnn2seqencoder : Lstm2SeqEncoder
            Lstm2SeqEncoder that encodes a set of instances to a sequence of hidden states
        encoding_dim : int
            Hidden dimension of the lstm2seq encoder
        namespace_to_constraints: Dict[str, List[Tuple[int, int]]]
            A set of constraints that are valid transitions
        include_start_end_trainsitions: bool
            Whether to include start end transitions
        """
        super(RnnSeqCrfTagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.encoding_dim = encoding_dim
        self.datasets_manager = datasets_manager

        self.label_namespaces = datasets_manager.label_namespaces
        self.device = device
        self.tagging_type = tagging_type
        self.crfs = nn.ModuleDict()
        self.linear_clfs = nn.ModuleDict()
        self.include_start_end_transitions = include_start_end_trainsitions

        if namespace_to_constraints is None and self.tagging_type is not None:
            namespace_to_constraints = defaultdict(list)
            for namespace in self.label_namespaces:
                idx2label_mapping = self.datasets_manager.get_idx_label_mapping(
                    label_namespace=namespace
                )
                transitions_allowed = allowed_transitions(
                    constraint_type=self.tagging_type, labels=idx2label_mapping
                )
                namespace_to_constraints[namespace] = transitions_allowed
        else:
            namespace_to_constraints = defaultdict(list)

        self.namespace_to_constraints = namespace_to_constraints
        for namespace in self.label_namespaces:
            num_labels = self.datasets_manager.num_labels[namespace]
            crf = CRF(
                num_tags=num_labels,
                constraints=self.namespace_to_constraints.get(namespace),
                include_start_end_transitions=self.include_start_end_transitions,
            )  # we do not add start and end tags to our labels
            clf = nn.Linear(self.encoding_dim, num_labels)
            self.crfs.update({namespace: crf})
            self.linear_clfs.update({namespace: clf})

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

        output_dict = {}
        for namespace in self.label_namespaces:
            # batch size, time steps, num_classes
            namespace_logits = self.linear_clfs[namespace](encoding)
            batch_size, time_steps, _ = namespace_logits.size()
            output_dict[f"logits_{namespace}"] = namespace_logits
            predicted_tags = self.crfs[namespace].viterbi_tags(
                logits=namespace_logits,
                mask=torch.ones(
                    size=(batch_size, time_steps), dtype=torch.long, device=self.device
                ),
            )
            predicted_tags = [tag for tag, _ in predicted_tags]
            output_dict[f"predicted_tags_{namespace}"] = predicted_tags

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
                    label_instances = torch.tensor(
                        label_instances, dtype=torch.long, device=self.device
                    )
                    labels_indices[namespace].append(label_instances)

            losses = []
            for namespace in self.label_namespaces:
                labels_tensor = labels_indices[namespace]
                labels_tensor = torch.stack(labels_tensor)
                batch_size, time_steps = labels_tensor.size()
                mask = torch.ones(
                    size=(batch_size, time_steps), dtype=torch.long, device=self.device
                )
                logits_namespace = output_dict[f"logits_{namespace}"]
                loss_ = -self.crfs[namespace](logits_namespace, labels_tensor, mask)
                losses.append(loss_)

            loss = sum(losses)
            output_dict["loss"] = loss

        return output_dict
