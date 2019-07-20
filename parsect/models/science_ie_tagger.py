import torch.nn as nn
from typing import Dict, Any, Union
from torchcrf import CRF
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder
import torch
import itertools


class ScienceIETagger(nn.Module):
    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        hid_dim: int,
        num_classes: int,
        character_encoder: Union[LSTM2VecEncoder, None] = None,
    ):
        super(ScienceIETagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.character_encoder = character_encoder

        self.task_crf = CRF(num_tags=self.num_classes, batch_first=True)
        self.process_crf = CRF(num_tags=self.num_classes, batch_first=True)
        self.material_crf = CRF(num_tags=self.num_classes, batch_first=True)

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
        tokens = iter_dict["tokens"]
        labels = iter_dict["label"]

        batch_size, max_time_steps = tokens.size()

        assert labels.ndimension() == 2, self.msg_printer(
            f"For Science IE Tagger, labels should have 2 dimensions"
            f"batch_size, 3 * max_length. The labels you passed have "
            f"{labels.ndimension()}"
        )

        task_labels, process_labels, material_labels = torch.chunk(
            labels, chunks=3, dim=1
        )

        character_encoding = None
        if self.character_encoder is not None:
            char_tokens = iter_dict["char_tokens"]
            char_tokens = char_tokens.view(batch_size * max_time_steps, -1)
            character_encoding = self.character_encoder(char_tokens)
            # batch size, time steps, hidden_dim
            character_encoding = character_encoding.view(batch_size, max_time_steps, -1)

        encoding = self.rnn2seqencoder(
            x=tokens, additional_embedding=character_encoding
        )

        # batch_size * time_steps * num_classes
        task_logits = self.hidden2task(encoding)
        process_logits = self.hidden2process(encoding)
        material_logits = self.hidden2material(encoding)

        assert task_logits.size(1) == process_logits.size(1) == material_logits.size(1)
        assert task_logits.size(2) == process_logits.size(2) == material_logits.size(2)

        predicted_task_tags = self.task_crf.decode(task_logits)  # List[List[int]] N * T
        predicted_process_tags = self.process_crf.decode(process_logits)
        predicted_material_tags = self.material_crf.decode(material_logits)

        assert (
            len(predicted_task_tags)
            == len(predicted_process_tags)
            == len(predicted_material_tags)
        )
        # arrange the labels in N * 3T
        zipped_tags = zip(
            predicted_task_tags, predicted_process_tags, predicted_material_tags
        )
        predicted_tags = itertools.chain.from_iterable(zipped_tags)
        predicted_tags = list(predicted_tags)

        output_dict = {
            "task_logits": task_logits,
            "process_logits": process_logits,
            "material_logits": material_logits,
            "predicted_task_tags": predicted_task_tags,
            "predicted_process_tags": predicted_process_tags,
            "predicted_material_tags": predicted_material_tags,
            "predicted_tags": predicted_tags,
        }

        if is_training or is_validation:
            task_loss = -self.task_crf(task_logits, task_labels)
            process_loss = -self.process_crf(process_logits, process_labels)
            material_loss = -self.material_crf(material_logits, material_labels)
            loss = task_loss + process_loss + material_loss
            output_dict["loss"] = loss

        return output_dict
