import torch.nn as nn
from typing import Dict, Any, Union
from torchcrf import CRF
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.lstm2vecencoder import LSTM2VecEncoder


class ParscitTagger(nn.Module):
    def __init__(
        self,
        rnn2seqencoder: Lstm2SeqEncoder,
        hid_dim: int,
        num_classes: int,
        character_encoder: Union[LSTM2VecEncoder, None] = None,
    ):
        """

        :param rnn2seqencoder: type: Lstm2SeqEncoder
        RNN2Seq encoder that encodes the sentences to hidden states
        :param hid_dim: type: int
        Hidden dimension of the rnn2seq encoder
        :param num_classes: type: int
        Num classes
        :param character_encoder: Lstm2VecEncoder
        Encoder lstm2vector that converts characters to word embeddings
        """
        super(ParscitTagger, self).__init__()

        self.rnn2seqencoder = rnn2seqencoder
        self.hidden_dim = hid_dim
        self.num_classes = num_classes
        self.character_encoder = character_encoder
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
        tokens = iter_dict["tokens"]

        batch_size, max_time_steps = tokens.size()

        character_encoding = None
        if self.character_encoder is not None:
            char_tokens = iter_dict["char_tokens"]
            char_tokens = char_tokens.view(batch_size * max_time_steps, -1)
            character_encoding = self.character_encoder(char_tokens)
            # batch size, time steps, hidden_dim
            character_encoding = character_encoding.view(batch_size, max_time_steps, -1)

        # batch size, time steps, hidden_dim
        encoding = self.rnn2seqencoder(
            x=tokens, additional_embedding=character_encoding
        )

        # batch size, time steps, num_classes
        logits = self.hidden2tag(encoding)

        predicted_tags = self.crf.decode(logits)

        output_dict = {"logits": logits, "predicted_tags": predicted_tags}

        if is_training or is_validation:
            labels = iter_dict["label"]
            assert labels.ndimension() == 2, self.msg_printer(
                f"For Parscit tagger, labels should have 2 dimensions. "
                f"You passed labels that have {labels.ndimension()}"
            )
            loss = -self.crf(logits, labels)
            output_dict["loss"] = loss

        return output_dict
