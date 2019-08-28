import spacy
from typing import List
from tqdm import tqdm
from wasabi import Printer
from sciwing.tokenizers.BaseTokenizer import BaseTokenizer
from sciwing.utils.custom_spacy_tokenizers import CustomSpacyWhiteSpaceTokenizer


class WordTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: str = "spacy"):
        super(WordTokenizer, self).__init__()
        self.msg_printer = Printer()
        self.tokenizer = tokenizer
        self.allowed_tokenizers = ["spacy", "nltk", "vanilla", "spacy-whitespace"]
        assert self.tokenizer in self.allowed_tokenizers, AssertionError(
            f"The word tokenizer can be {self.allowed_tokenizers}"
        )

        if self.tokenizer == "spacy" or "spacy-whitespace":
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.remove_pipe("parser")
            self.nlp.remove_pipe("tagger")
            self.nlp.remove_pipe("ner")

        if self.tokenizer == "spacy-whitespace":
            self.nlp.tokenizer = CustomSpacyWhiteSpaceTokenizer(self.nlp.vocab)

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer == "spacy" or self.tokenizer == "spacy-whitespace":
            doc = self.nlp(text)
            tokens = [
                token.text for token in doc if bool(token.text.strip())
            ]  # add token text only if they are not empty
            return tokens

        if self.tokenizer == "vanilla":
            tokens = text.split()
            return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        tokenized = []
        for text in texts:
            tokenized.append(self.tokenize(text))
        return tokenized
