from parsect.vocab.vocab import Vocab
import os
import parsect.constants as constants
from typing import Dict
import numpy as np
from tqdm import tqdm

PATHS = constants.PATHS
DATA_DIR = PATHS['DATA_DIR']


class WordEmbLoader:
    """
    This handles the loading of word embeddings for a vocab
    This can handle different kinds of embeddings.

    """
    def __init__(self,
                 vocab: Vocab,
                 embedding_type: str = 'glove_6B'):
        """

        :param vocab: type: Vocab
        All the words in the vocab are considered for loading
        :param embedding_type: type: List
        """
        self.vocab = vocab
        self.token2idx_mapping = self.vocab.get_token2idx_mapping()
        self.embedding_type = embedding_type
        self.allowed_embedding_types = ['glove_6B_50', 'glove_6B_100', 'glove_6B_200', 'glove_6B_300']

        assert self.embedding_type in self.allowed_embedding_types, \
            "You can use one of {0} for embedding type".format(self.allowed_embedding_types)

        self.embedding_filename = self.get_preloaded_filename()
        self.vocab_embedding = {}  # stores the embedding for all words in vocab

        if 'glove' in self.embedding_type:
            self.vocab_embedding = self.load_glove_embedding()

    def get_preloaded_filename(self):
        filename = None

        if self.embedding_type == 'glove_6B_50':
            filename = os.path.join(DATA_DIR, 'embeddings', 'glove', 'glove.6B.50d.txt')

        elif self.embedding_type == 'glove_6B_100':
            filename = os.path.join(DATA_DIR, 'embeddings', 'glove', 'glove.6B.100d.txt')

        elif self.embedding_type == 'glove_6B_200':
            filename = os.path.join(DATA_DIR, 'embeddings', 'glove', 'glove.6B.200d.txt')

        elif self.embedding_type == 'glove_6B_300':
            filename = os.path.join(DATA_DIR, 'embeddings', 'glove', 'glove.6B.300d.txt')

        return filename

    def load_glove_embedding(self) -> Dict[str, np.array]:
        """
        Imports the glove embedding
        Loads the word embedding for words in the vocabulary
        If the word in the vocabulary doesnot have an embedding
        then it is loaded with zeros
        TODO: Load only once in the project and store it in json file
            - Read from json file at once
            - This might be memory expensive and save a little bit of time
        :return:
        """
        embedding_dim = int(self.embedding_type.split('_')[-1])
        glove_embeddings = {}
        with open(self.embedding_filename, 'r') as fp:
            for line in tqdm(fp, desc="Loading embeddings from file {0}".format(
                self.embedding_type
            )):
                values = line.split()
                word = values[0]
                embedding = np.array([float(value) for value in values[1:]])
                glove_embeddings[word] = embedding

        tokens = self.token2idx_mapping.keys()

        vocab_embeddings = {}

        for token in tokens:
            try:
                 emb = glove_embeddings[token]
            except KeyError:
                emb = np.zeros(embedding_dim)

            vocab_embeddings[token] = emb

        return vocab_embeddings


if __name__ == '__main__':
    vocab = Vocab(
        instances=[['i', 'like', 'nlp'], ['i', 'dont', 'like', 'bs']],
        max_num_words=1000
    )
    vocab.build_vocab()
    word_emb_loader = WordEmbLoader(vocab=vocab,
                                    embedding_type='glove_6B_50')


