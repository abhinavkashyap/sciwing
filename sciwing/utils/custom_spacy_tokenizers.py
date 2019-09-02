"""
This module implements custom spacy tokenizers if needed
This can be useful for custom tokenization that is required for scientific domain
"""
from spacy.tokens import Doc


class CustomSpacyWhiteSpaceTokenizer(object):
    def __init__(self, vocab):
        """ White space tokenizer tokenizes the word according to spaces.

        Parameters
        ----------
        vocab : nlp.vocab
            Spacy vocab object
        """
        self.vocab = vocab

    def __call__(self, text):
        """
        Parameters
        ----------
        text : str
            The text that should be tokenized

        Returns
        -------
        spacy.tokens.Doc
            Spacy docment with the tokenized text and information what is a space.


        """
        words = text.split(" ")
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
