import torch
import torch.nn as nn
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizers.transformer_numericalizer import NumericalizerForTransformer
from sciwing.modules.embedders.base_embedders import BaseEmbedder
from sciwing.data.datasets_manager import DatasetsManager
from typing import List, Union
import wasabi
import sciwing.constants as constants
from pytorch_pretrained_bert import BertModel
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
import os

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertEmbedder(nn.Module, BaseEmbedder, ClassNursery):
    def __init__(
        self,
        datasets_manager: DatasetsManager = None,
        dropout_value: float = 0.0,
        aggregation_type: str = "sum",
        bert_type: str = "bert-base-uncased",
        word_tokens_namespace="tokens",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Bert Embedder that embeds the given instance to BERT embeddings

        Parameters
        ----------
        dropout_value : float
            The amount of dropout to be added after the embedding
        aggregation_type : str
            The kind of aggregation of different layers. BERT produces representations from
            different layers. This specifies the strategy to aggregating them
            One of

            sum
                Sum the representations from all the layers
            average
                Average the representations from all the layers

        bert_type : type
            The kind of BERT embedding to be used

            bert-base-uncased
                12 layer transformer trained on lowercased vocab

            bert-large-uncased:
                24 layer transformer trained on lowercased vocab

            bert-base-cased:
                12 layer transformer trained on cased vocab

            bert-large-cased:
                24 layer transformer train on cased vocab

            scibert-base-cased
                12 layer transformer trained on scientific document on cased normal vocab
            scibert-sci-cased
                12 layer transformer trained on scientific documents on cased scientifc vocab

            scibert-base-uncased
                12 layer transformer trained on scientific docments on uncased normal vocab

            scibert-sci-uncased
                12 layer transformer train on scientific documents on ncased scientific vocab

        word_tokens_namespace : str
            The namespace in the liens where the tokens are stored

        device :  Union[torch.device, str]
            The device on which the model is run.
        """
        super(BertEmbedder, self).__init__()

        self.datasets_manager = datasets_manager
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.bert_type = bert_type
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.word_tokens_namespace = word_tokens_namespace
        self.msg_printer = wasabi.Printer()
        self.embedder_name = bert_type

        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }

        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = os.path.join(
                EMBEDDING_CACHE_DIR, foldername, "weights.tar.gz"
            )

        else:
            self.model_type_or_folder_url = self.bert_type

        # load the bert model
        with self.msg_printer.loading(" Loading Bert tokenizer and model. "):
            self.bert_tokenizer = TokenizerForBert(
                bert_type=self.bert_type, do_basic_tokenize=False
            )
            self.bert_numericalizer = NumericalizerForTransformer(
                tokenizer=self.bert_tokenizer
            )
            self.model = BertModel.from_pretrained(self.model_type_or_folder_url)
            self.model.eval()
            self.model.to(self.device)

        self.msg_printer.good(f"Finished Loading {self.bert_type} model and tokenizer")
        self.embedding_dimension = self.get_embedding_dimension()

    def forward(self, lines: List[Line]) -> torch.Tensor:
        """

        Parameters
        ----------
        lines : List[Line]
            A list of lines

        Returns
        -------
        torch.Tensor
            The bert embeddings for all the words in the instances
            The size of the returned embedding is ``[batch_size, max_len_word_tokens, emb_dim]``

        """

        # word_tokenize all the text string in the batch
        bert_tokens_lengths = []
        word_tokens_lengths = []
        for line in lines:
            text = line.text
            word_tokens = line.tokens[self.word_tokens_namespace]
            word_tokens_lengths.append(len(word_tokens))

            # split every token to subtokens
            for word_token in word_tokens:
                word_piece_tokens = self.bert_tokenizer.tokenize(word_token.text)
                word_token.sub_tokens = word_piece_tokens

            bert_tokenized_text = self.bert_tokenizer.tokenize(text)
            line.tokenizers[self.embedder_name] = self.bert_tokenizer
            line.add_tokens(tokens=bert_tokenized_text, namespace=self.embedder_name)
            bert_tokens_lengths.append(len(bert_tokenized_text))

        max_len_bert = max(bert_tokens_lengths)
        max_len_words = max(word_tokens_lengths)
        # pad the tokenized text to a maximum length
        indexed_tokens = []
        segment_ids = []
        for line in lines:
            bert_tokens = line.tokens[self.embedder_name]
            tokens_numericalized = self.bert_numericalizer.numericalize_instance(
                instance=bert_tokens
            )
            tokens_numericalized = self.bert_numericalizer.pad_instance(
                numericalized_text=tokens_numericalized,
                max_length=max_len_bert + 2,
                add_start_end_token=True,
            )
            segment_numbers = [0] * len(tokens_numericalized)

            tokens_numericalized = torch.LongTensor(tokens_numericalized)
            segment_numbers = torch.LongTensor(segment_numbers)

            indexed_tokens.append(tokens_numericalized)
            segment_ids.append(segment_numbers)

        tokens_tensor = torch.stack(indexed_tokens)
        segment_tensor = torch.stack(segment_ids)

        tokens_tensor = tokens_tensor.to(self.device)
        segment_tensor = segment_tensor.to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segment_tensor)

        if "base" in self.bert_type:
            assert len(encoded_layers) == 12
        elif "large" in self.bert_type:
            assert len(encoded_layers) == 24

        # num_bert_layers, batch_size, max_len_bert + 2, bert_hidden_dimension
        all_layers = torch.stack(encoded_layers, dim=0)

        # batch_size, max_len_bert + 2, bert_hidden_dimension
        if self.aggregation_type == "sum":
            encoding = torch.sum(all_layers, dim=0)

        elif self.aggregation_type == "average":
            encoding = torch.mean(all_layers, dim=0)
        else:
            raise ValueError(f"The aggregation type {self.aggregation_type}")

        # fill up the appropriate embeddings in the tokens of the lines
        batch_embeddings = []
        for idx, line in enumerate(lines):
            word_tokens = line.tokens[self.word_tokens_namespace]  # word tokens
            bert_tokens_ = line.tokens[self.embedder_name]
            token_embeddings = encoding[idx]  # max_len_bert + 2, bert_hidden_dimensiofn

            len_word_tokens = len(word_tokens)
            len_bert_tokens = len(bert_tokens_)
            padding_length_bert = max_len_bert - len_bert_tokens
            padding_length_words = max_len_words - len_word_tokens

            # do not want embeddings for padding
            if padding_length_bert > 0:
                token_embeddings = token_embeddings[:-padding_length_bert]

            # do not want embeddings for start and end tokens
            token_embeddings = token_embeddings[1:-1]

            # just have embeddings for the bert tokens now
            # without padding and start and end tokens
            assert token_embeddings.size(0) == len_bert_tokens, (
                f"bert token embeddings size {token_embeddings.size()} and length of bert tokens "
                f"{len_bert_tokens}"
            )

            line_embeddings = []
            for token in word_tokens:
                idx = 0
                sub_tokens = token.sub_tokens
                len_sub_tokens = len(sub_tokens)

                # taking the embedding of only the first token
                # TODO: Have different strategies for this
                emb = token_embeddings[idx]
                line_embeddings.append(emb)
                token.set_embedding(name=self.embedder_name, value=emb)
                idx += len_sub_tokens

            for i in range(padding_length_words):
                zeros = torch.zeros(self.embedding_dimension)
                zeros = zeros.to(self.device)
                line_embeddings.append(zeros)

            line_embeddings = torch.stack(line_embeddings)
            batch_embeddings.append(line_embeddings)

        # batch_size, max_len_words, bert_hidden_dimension
        batch_embeddings = torch.stack(batch_embeddings)
        return batch_embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size
