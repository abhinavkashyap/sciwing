import spacy
from typing import List
from tqdm import tqdm
from wasabi import Printer
from parsect.tokenizers.BaseTokenizer import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    The tokenizer, takes a span of text and returns
    the tokens (it can be either word or character)
    """

    def __init__(self, tokenizer: str = "spacy"):
        """

        :param tokenizer: type:str
        We can use either of ['spacy'] tokenizer to word_tokenize text
        """
        super(WordTokenizer, self).__init__()
        self.msg_printer = Printer()
        self.tokenizer = tokenizer
        self.allowed_tokenizers = ["spacy", "nltk", "vanilla"]
        assert self.tokenizer in self.allowed_tokenizers, AssertionError(
            "The word tokenizer can be either spacy or nltk"
        )

        if self.tokenizer == "spacy":
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.remove_pipe("parser")
            self.nlp.remove_pipe("tagger")
            self.nlp.remove_pipe("ner")

    def tokenize(self, text: str) -> List[str]:
        """

        :param text: type: str
        Span of text to be word tokenized
        :return:
        """

        if self.tokenizer == "spacy":
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
        for text in tqdm(texts, total=len(texts), desc="Batch word_tokenize"):
            tokenized.append(self.tokenize(text))

        self.msg_printer.good("Finished Tokenizing Text")
        return tokenized
