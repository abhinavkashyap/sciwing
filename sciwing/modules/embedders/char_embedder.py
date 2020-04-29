import torch.nn as nn
from typing import List, Union
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.data.line import Line
import torch


class CharEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        char_embedding_dimension: int,
        hidden_dimension: int,
        datasets_manager: DatasetsManager = None,
        word_tokens_namespace: str = "tokens",
        char_tokens_namespace: str = "char_tokens",
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """ This is a character embedder that takes in lines and collates the character
        embeddings for all the tokens in the lines.

        Parameters
        ----------
        char_embedding_dimension : int
            The dimension of the character embedding
        word_tokens_namespace : int
            The name space where the words are saved
        char_tokens_namespace : str
            The namespace where the character tokens are saved
        datasets_manager : DatasetsManager
            The dataset manager that handles all the datasets
        hidden_dimension : int
            The hidden dimension of the LSTM which will be used to get
            character embeddings
        """
        super(CharEmbedder, self).__init__()
        self.char_embedding_dimension = char_embedding_dimension
        self.word_tokens_namespace = word_tokens_namespace
        self.char_tokens_namespace = char_tokens_namespace
        self.datasets_manager = datasets_manager
        self.hidden_dimension = hidden_dimension
        self.device = torch.device(device) if isinstance(device, str) else device

        self.char_vocab = self.datasets_manager.namespace_to_vocab[
            self.char_tokens_namespace
        ]
        self.word_vocab = self.datasets_manager.namespace_to_vocab[
            self.word_tokens_namespace
        ]

        self.char_numericalizer = self.datasets_manager.namespace_to_numericalizer[
            self.char_tokens_namespace
        ]
        self.word_numericalizer = self.datasets_manager.namespace_to_numericalizer[
            self.word_tokens_namespace
        ]
        self.idx2items = self.char_vocab.idx2token
        self.num_embeddings = len(self.idx2items)
        self.embedder_name = "char_embedding"

        self.embedding = nn.Embedding(
            self.num_embeddings, self.char_embedding_dimension
        )
        nn.init.xavier_normal_(self.embedding.weight)

        self.char_rnn = nn.LSTM(
            self.char_embedding_dimension,
            self.hidden_dimension,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.embedding_dimension = self.get_embedding_dimension()

    def forward(self, lines: List[Line]):
        batch_size = len(lines)
        token_lengths = []
        line_lengths = []

        # find the maximum token length
        for line in lines:
            word_tokens = line.tokens[self.word_tokens_namespace]
            len_tokens = [len(token.text) for token in word_tokens]
            line_lengths.append(len(word_tokens))
            token_lengths.extend(len_tokens)

        max_token_length = max(token_lengths)
        max_line_length = max(line_lengths)

        # numericalized version of all the characters in the lines
        batch_numericalized = []  # batch_size * max_line_length * max_token_length
        for line in lines:
            word_tokens = line.tokens[self.word_tokens_namespace]
            word_tokens = [tok.text for tok in word_tokens]
            word_tokens_num = self.word_numericalizer.numericalize_instance(
                instance=word_tokens
            )
            word_tokens_num_padded = self.word_numericalizer.pad_instance(
                numericalized_text=word_tokens_num,
                max_length=max_line_length,
                add_start_end_token=False,
            )
            word_tokens_padded = [
                self.word_vocab.get_token_from_idx(token)
                for token in word_tokens_num_padded
            ]

            line_numericalized = []
            for token in word_tokens_padded:
                char_tokens = [char for char in token]
                char_numericalized = self.char_numericalizer.numericalize_instance(
                    char_tokens
                )
                char_numericalized = self.char_numericalizer.pad_instance(
                    numericalized_text=char_numericalized,
                    max_length=max_token_length,
                    add_start_end_token=False,
                )  # max_num_chars
                char_numericalized = torch.tensor(
                    char_numericalized, device=self.device, dtype=torch.long
                )
                line_numericalized.append(char_numericalized)
            line_numericalized = torch.stack(
                line_numericalized
            )  # max_line_length * max_num_chars
            batch_numericalized.append(line_numericalized)

        batch_numericalized = torch.stack(batch_numericalized)
        batch_numericalized = batch_numericalized.view(
            batch_size * max_line_length, max_token_length
        )

        # get embedding for every character

        # batch_size * max_line_length, max_token_length, char_emb_dim
        embedded_tokens = self.embedding(batch_numericalized)

        # pass through bilstm

        # output: batch_size * max_line_length, max_token_length, num_directions * hidden_size
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (h_n, c_n) = self.char_rnn(embedded_tokens)

        # concat forward and backward hidden states
        forward_hidden = h_n[0, :, :]
        backward_hidden = h_n[1, :, :]
        encoding = torch.cat([forward_hidden, backward_hidden], dim=1)
        encoding = encoding.view(
            batch_size, max_line_length, -1
        )  # batch_size, max_line_length, embedding_dimension

        # set the character embeddings in the line tokens
        for idx, line in enumerate(lines):
            line_tokens = line.tokens[self.word_tokens_namespace]
            line_embeddings = encoding[idx]

            # note: line_tokens has no padding tokens
            # the zip will get the embeddings for non pad tokens here
            for token, embedding in zip(line_tokens, line_embeddings):
                token.set_embedding(self.embedder_name, embedding)

        return encoding

    def get_embedding_dimension(self) -> int:
        return self.hidden_dimension * 2
