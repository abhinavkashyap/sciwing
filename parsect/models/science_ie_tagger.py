import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
import torch
import copy
from parsect.utils.tensor_utils import get_mask
from parsect.utils.class_nursery import ClassNursery


class ScienceIETagger(nn.Module, ClassNursery):
    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        hid_dim: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
        task_constraints: Optional[List[Tuple[int, int]]] = None,
        process_constraints: Optional[List[Tuple[int, int]]] = None,
        material_constraints: Optional[List[Tuple[int, int]]] = None,
        character_encoder: Optional[LSTM2VecEncoder] = None,
        include_start_end_transitions: Optional[bool] = False,
    ):
        super(ScienceIETagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.device = device
        self._task_constraints = task_constraints
        self._process_constraints = process_constraints
        self._material_constraints = material_constraints
        self.character_encoder = character_encoder
        self.include_start_end_transitions = include_start_end_transitions

        self.task_crf = CRF(
            num_tags=self.num_classes,
            constraints=task_constraints,
            include_start_end_transitions=include_start_end_transitions,
        )
        self.process_crf = CRF(
            num_tags=self.num_classes,
            constraints=process_constraints,
            include_start_end_transitions=include_start_end_transitions,
        )
        self.material_crf = CRF(
            num_tags=self.num_classes,
            constraints=material_constraints,
            include_start_end_transitions=include_start_end_transitions,
        )

        self.hidden2task = nn.Linear(self.hid_dim, self.num_classes)
        self.hidden2process = nn.Linear(self.hid_dim, self.num_classes)
        self.hidden2material = nn.Linear(self.hid_dim, self.num_classes)

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
        encoding = self.rnn2seqencoder(iter_dict=iter_dict)

        # batch_size * time_steps * num_classes
        task_logits = self.hidden2task(encoding)
        process_logits = self.hidden2process(encoding)
        material_logits = self.hidden2material(encoding)

        batch_size, time_steps, _ = task_logits.size()
        mask = torch.ones(size=(batch_size, time_steps), dtype=torch.long)
        mask = torch.LongTensor(mask)
        mask = mask.to(self.device)

        assert task_logits.size(1) == process_logits.size(1) == material_logits.size(1)
        assert task_logits.size(2) == process_logits.size(2) == material_logits.size(2)

        # List[List[int]] N * T
        predicted_task_tags = self.task_crf.viterbi_tags(logits=task_logits, mask=mask)
        predicted_process_tags = self.process_crf.viterbi_tags(
            logits=process_logits, mask=mask
        )
        predicted_material_tags = self.material_crf.viterbi_tags(
            logits=material_logits, mask=mask
        )

        predicted_task_tags = [tag for tag, _ in predicted_task_tags]
        predicted_process_tags = [tag for tag, _ in predicted_process_tags]
        predicted_material_tags = [tag for tag, _ in predicted_material_tags]

        # add the appropriate numbers
        predicted_task_tags = torch.LongTensor(predicted_task_tags)
        predicted_process_tags = torch.LongTensor(predicted_process_tags) + 8
        predicted_material_tags = torch.LongTensor(predicted_material_tags) + 16

        assert (
            len(predicted_task_tags)
            == len(predicted_process_tags)
            == len(predicted_material_tags)
        )
        # arrange the labels in N * 3T
        predicted_tags = torch.cat(
            [predicted_task_tags, predicted_process_tags, predicted_material_tags],
            dim=1,
        )
        predicted_tags = predicted_tags.tolist()

        output_dict = {
            "task_logits": task_logits,
            "process_logits": process_logits,
            "material_logits": material_logits,
            "predicted_task_tags": predicted_task_tags.tolist(),
            "predicted_process_tags": predicted_process_tags.tolist(),
            "predicted_material_tags": predicted_material_tags.tolist(),
            "predicted_tags": predicted_tags,
        }

        if is_training or is_validation:
            labels = iter_dict["label"]
            len_tokens = iter_dict["len_tokens"]
            mask = get_mask(
                batch_size=batch_size, max_size=time_steps, lengths=len_tokens
            )
            mask = mask.to(self.device)
            # if you change label then iter_dict["label"] gets screwed
            labels_copy = copy.deepcopy(labels)
            assert labels.ndimension() == 2, self.msg_printer(
                f"For Science IE Tagger, labels should have 2 dimensions"
                f"batch_size, 3 * max_length. The labels you passed have "
                f"{labels.ndimension()}"
            )

            task_labels, process_labels, material_labels = torch.chunk(
                labels_copy, chunks=3, dim=1
            )
            process_labels -= 8
            material_labels -= 16
            task_loss = -self.task_crf(task_logits, task_labels, mask)
            process_loss = -self.process_crf(process_logits, process_labels, mask)
            material_loss = -self.material_crf(material_logits, material_labels, mask)
            loss = task_loss + process_loss + material_loss
            output_dict["loss"] = loss

        return output_dict
