import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from parsect.utils.common import pack_to_length
from typing import List
import wasabi
import parsect.constants as constants
import os

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]


class BowBertEncoder:
    def __init__(
        self,
        emb_dim: int = 768,
        dropout_value: float = 0.0,
        aggregation_type: str = "sum",
        bert_type: str = "bert-base-uncased",
    ):
        """

        :param emb_dim: type: int
        Embedding dimension for bert
        :param dropout_value: type: float
        Dropout value that can be used for embedding
        :param aggregation_type: type: str
        sum - sums the embeddings of tokens in an instance
        average - averages the embedding of tokens in an instance
        In the case of bert, we also take sum or average of all the different layers
        :param bert_type: type: str
        There are different bert models
        bert-base-uncased - 12 layer 768 hidden dimensional
        bert-large-uncased 24 layer 1024 hidden dimensional
        """
        super(BowBertEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.bert_type = bert_type
        self.msg_printer = wasabi.Printer()
        self.allowed_bert_types = [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "scibert-base-cased",
            "scibert-sci-cased",
            "scibert-base-uncased",
            "scibert-sci-uncased"
        ]
        self.scibert_foldername_mapping = {
            'scibert-base-cased': 'scibert_basevocab_cased',
            'scibert-sci-cased': 'scibert_scivocab_cased',
            'scibert-base-uncased': 'scibert_basevocab_uncased',
            'scibert-sci-uncased': 'scibert_scivocab_uncased'
        }
        self.model_type_or_folder_url = None
        self.vocab_type_or_filename = None

        assert self.bert_type in self.allowed_bert_types

        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = os.path.join(MODELS_CACHE_DIR, foldername, 'weights.tar.gz')
            self.vocab_type_or_filename = os.path.join(MODELS_CACHE_DIR,foldername, 'vocab.txt')
        else:
            self.model_type_or_folder_url = self.bert_type
            self.vocab_type_or_filename = self.bert_type

        # load the bert model
        with self.msg_printer.loading("Loading Bert tokenizer and model"):
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.vocab_type_or_filename)
            self.model = BertModel.from_pretrained(self.model_type_or_folder_url)
            self.model.eval()

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))

        self.msg_printer.good(f"Finished Loading {self.bert_type} model and tokenizer")

    def forward(self, x: List[str]) -> torch.Tensor:

        # tokenize all the text string in the batch
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

        # TODO: patched .. need to change it
        tokens_tensor = tokens_tensor.to(torch.device("cuda:0")) if torch.cuda.is_available() else tokens_tensor
        segment_tensor = segment_tensor.to(torch.device("cuda:0")) if torch.cuda.is_available() else segment_tensor


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
            sum_instances = torch.sum(sum_layers, dim=1)
            return sum_instances

        elif self.aggregation_type == "average":
            average_layers = torch.mean(all_layers, dim=0)
            average_instances = torch.mean(average_layers, dim=1)
            return average_instances

    def __call__(self, x: List[str]) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    strings = [
        "Lets start by talking politics",
        "there are radical ways to test your code",
    ]
    emb_dim = 768
    dropout_value = 0.0
    aggregation_type = "average"
    bert_type = "bert-base-cased"

    bow_bert_encoder = BowBertEncoder(
        emb_dim=emb_dim,
        dropout_value=dropout_value,
        aggregation_type=aggregation_type,
        bert_type=bert_type,
    )

    bow_bert_encoder(strings)
