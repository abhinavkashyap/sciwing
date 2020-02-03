from pytorch_pretrained_bert import BertTokenizer
import wasabi
import os
import sciwing.constants as constants
from typing import List
from tqdm import tqdm
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class TokenizerForBert(BaseTokenizer):
    def __init__(self, bert_type: str, do_basic_tokenize=True):
        super(TokenizerForBert, self).__init__()
        self.bert_type = bert_type
        self.do_basic_tokenize = do_basic_tokenize
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
        assert bert_type in self.allowed_bert_types, self.msg_printer.fail(
            f"You passed {bert_type} for attribute bert_type."
            f"The allowed types are {self.allowed_bert_types}"
        )
        self.vocab_type_or_filename = None
        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.vocab_type_or_filename = os.path.join(
                EMBEDDING_CACHE_DIR, foldername, "vocab.txt"
            )
        else:
            self.vocab_type_or_filename = self.bert_type

        with self.msg_printer.loading("Loading Bert model"):
            self.tokenizer = BertTokenizer.from_pretrained(
                self.vocab_type_or_filename, do_basic_tokenize=do_basic_tokenize
            )

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        tokenized = []
        for text in tqdm(texts, total=len(texts), desc=f"Bert tokenizing"):
            tokenized.append(self.tokenize(text))

        self.msg_printer.good(f"Finished tokenizing text")
        return tokenized
