import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sciwing.utils.common import pack_to_length
from typing import Dict, Any, Union
import wasabi
import sciwing.constants as constants
import os
from sciwing.utils.class_nursery import ClassNursery

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertEmbedder(nn.Module, ClassNursery):
    def __init__(
        self,
        emb_dim: int = 768,
        dropout_value: float = 0.0,
        aggregation_type: str = "sum",
        bert_type: str = "bert-base-uncased",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Bert Embedder that embeds the given instance to BERT embeddings

        Parameters
        ----------
        emb_dim : int
            Embedding dimension
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

        device :  Union[torch.device, str]
            The device on which the model is run.
        """
        super(BertEmbedder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.bert_type = bert_type
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.msg_printer = wasabi.Printer()
        self.allowed_bert_types = [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "scibert-base-cased",
            "scibert-sci-cased",
            "scibert-base-uncased",
            "scibert-sci-uncased",
        ]
        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }
        self.model_type_or_folder_url = None
        self.vocab_type_or_filename = None

        assert self.bert_type in self.allowed_bert_types

        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = os.path.join(
                EMBEDDING_CACHE_DIR, foldername, "weights.tar.gz"
            )
            self.vocab_type_or_filename = os.path.join(
                EMBEDDING_CACHE_DIR, foldername, "vocab.txt"
            )
        else:
            self.model_type_or_folder_url = self.bert_type
            self.vocab_type_or_filename = self.bert_type

        # load the bert model
        with self.msg_printer.loading(" Loading Bert tokenizer and model. "):
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.vocab_type_or_filename
            )
            self.model = BertModel.from_pretrained(self.model_type_or_folder_url)
            self.model.eval()
            self.model.to(self.device)

        self.msg_printer.good(f"Finished Loading {self.bert_type} model and tokenizer")

    def forward(self, iter_dict: Dict[str, Any]) -> torch.Tensor:
        """

        Parameters
        ----------
        iter_dict : Dict[str, Any]
            It expects "raw_intance" to be present in the iter dict.
            "raw_instance" is the instance that is not padded

        Returns
        -------
        torch.Tensor
            The bert embeddings for all the words in the instances
            The size of the returned embedding is ``[batch_size, num_time_steps, emb_dim]``

        """

        # word_tokenize all the text string in the batch
        x = iter_dict["raw_instance"]
        tokenized_text = list(map(self.bert_tokenizer.tokenize, x))
        lengths = list(map(lambda tokenized: len(tokenized), tokenized_text))
        max_len = sorted(lengths, reverse=True)[0]

        # pad the tokenized text to a maximum length
        padded_tokenized_text = []
        for tokens in tokenized_text:
            padded_tokens = pack_to_length(
                tokenized_text=tokens,
                max_length=max_len,
                pad_token="[PAD]",
                add_start_end_token=True,
                start_token="[CLS]",
                end_token="[SEP]",
            )
            padded_tokenized_text.append(padded_tokens)

        # convert them to ids based on bert vocab
        indexed_tokens = list(
            map(self.bert_tokenizer.convert_tokens_to_ids, padded_tokenized_text)
        )
        segment_ids = list(
            map(lambda tokens_list: [0] * len(tokens_list), indexed_tokens)
        )

        tokens_tensor = torch.tensor(indexed_tokens)
        segment_tensor = torch.tensor(segment_ids)

        tokens_tensor = tokens_tensor.to(self.device)
        segment_tensor = segment_tensor.to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segment_tensor)

        if "base" in self.bert_type:
            assert len(encoded_layers) == 12
        elif "large" in self.bert_type:
            assert len(encoded_layers) == 24

        # num_bert_layers, batch_size, sequence_length, bert_hidden_dimension
        all_layers = torch.stack(encoded_layers, dim=0)

        if self.aggregation_type == "sum":
            sum_layers = torch.sum(all_layers, dim=0)
            return sum_layers

        elif self.aggregation_type == "average":
            average_layers = torch.mean(all_layers, dim=0)
            return average_layers

    def __call__(self, iter_dict: Dict[str, Any]) -> torch.Tensor:
        return self.forward(iter_dict)
