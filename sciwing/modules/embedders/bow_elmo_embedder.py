import torch
from allennlp.commands.elmo import ElmoEmbedder
import wasabi
from typing import List, Union
import torch.nn as nn
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.modules.embedders.base_embedders import BaseEmbedder


class BowElmoEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        datasets_manager: DatasetsManager = None,
        layer_aggregation: str = "sum",
        device: Union[str, torch.device] = torch.device("cpu"),
        word_tokens_namespace="tokens",
    ):
        """ Bag of words Elmo Embedder which aggregates elmo embedding for every token

        Parameters
        ----------
        layer_aggregation : str
            You can chose one of ``[sum, average, last, first]``
            which decides how to aggregate different layers of ELMO. ELMO produces three
            layers of representations

            sum
                Representations from different layers are summed
            average
                Representations from different layers are average
            last
                Representations from last layer is considered
            first
                Representations from first layer is considered

        device : Union[str, torch.device]
            device for running the model on

        word_tokens_namespace: int
            Namespace where all the word tokens are stored
        """
        super(BowElmoEmbedder, self).__init__()
        self.dataset_manager = datasets_manager
        self.embedding_dimension = self.get_embedding_dimension()
        self.embedder_name = "elmo"
        self.word_tokens_namespace = word_tokens_namespace
        self.layer_aggregation_type = layer_aggregation
        self.allowed_layer_aggregation_types = ["sum", "average", "last", "first"]
        self.device = (
            torch.device(device) if isinstance(device, str) else torch.device(device)
        )

        if self.device.index:
            self.cuda_device_id = self.device.index
        else:
            self.cuda_device_id = -1
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

    def forward(self, lines: List[Line]) -> torch.Tensor:
        """

        Parameters
        ----------
        lines : List[Line]
            Just a list of lines

        Returns
        -------
        torch.Tensor
            Returns the representation for every token in the instance
            ``[batch_size, max_num_words, emb_dim]``. In case of Elmo the ``emb_dim`` is 1024


        """
        # [np.array] - A generator of embeddings
        # each array in the list is of the shape (3, #words_in_sentence, 1024)

        batch_tokens = []
        token_lengths = []
        for line in lines:
            line_tokens = line.tokens[self.word_tokens_namespace]
            line_tokens = [tok.text for tok in line_tokens]
            batch_tokens.append(line_tokens)
            token_lengths.append(len(line_tokens))

        max_len = max(token_lengths)
        embedded = list(self.elmo.embed_sentences(batch_tokens))

        batch_embeddings = []

        for idx, (line, embedding) in enumerate(zip(lines, embedded)):
            tokens = line.tokens[self.word_tokens_namespace]
            line_embeddings = []
            padding_length = max_len - len(tokens)
            embedding = torch.FloatTensor(embedding)
            embedding = embedding.to(self.device)

            # 3, #words_in_sentence, 1024

            # aggregate of word embeddings
            if self.layer_aggregation_type == "sum":
                # words_in_sentence, 1024
                embedding = torch.sum(embedding, dim=0)

            elif self.layer_aggregation_type == "average":
                # mean across all layers
                embedding = torch.mean(embedding, dim=0)

            elif self.layer_aggregation_type == "last":
                # words_in_sentence, 1024
                embedding = embedding[-1, :, :]

            elif self.layer_aggregation_type == "first":
                # words_in_sentence, 1024
                embedding = embedding[0, :, :]
            else:
                raise ValueError(
                    f"Layer aggregation can be one of sum, average, last and first"
                )

            for token, token_emb in zip(tokens, embedding):
                token.set_embedding(self.embedder_name, token_emb)
                line_embeddings.append(token_emb)

            # for batching
            for i in range(padding_length):
                zeros = torch.zeros(self.embedding_dimension, device=self.device)
                line_embeddings.append(zeros)

            line_embeddings = torch.stack(line_embeddings)
            batch_embeddings.append(line_embeddings)

        batch_embeddings = torch.stack(batch_embeddings)
        return batch_embeddings

    def get_embedding_dimension(self) -> int:
        return 1024
