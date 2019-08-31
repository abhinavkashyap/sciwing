import torch.nn as nn
from typing import Dict, Any
from torchcrf import CRF
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder


class ParscitTagger(nn.Module):
    """PyTorch module for Neural Parscit"""

    def __init__(self, rnn2seqencoder: Lstm2SeqEncoder, hid_dim: int, num_classes: int):
        """

        Parameters
        ----------
        rnn2seqencoder : Lstm2SeqEncoder
            Lstm2SeqEncoder that encodes a set of instances to a sequence of hidden states
        hid_dim : int
            Hidden dimension of the lstm2seq encoder
        num_classes : int
            The number of classes for every token
        """
        super(ParscitTagger, self).__init__()
        self.rnn2seqencoder = rnn2seqencoder
        self.hidden_dim = hid_dim
        self.num_classes = num_classes
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(
        self,
        iter_dict: Dict[str, Any],
        is_training: bool,
        is_validation: bool,
        is_test: bool,
    ):
        """
        Parameters
        ----------
        iter_dict : Dict[str, Any]
            ``iter_dict`` from any dataset that will be passed on to the encoder
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

        # batch size, time steps, hidden_dim
        encoding = self.rnn2seqencoder(iter_dict=iter_dict)

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
