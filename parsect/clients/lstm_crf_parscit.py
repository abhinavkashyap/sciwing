"""
This will directly use a allennlp crf tagger
"""
import torch
import torch.nn as nn


from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.dataset_readers import Conll2003DatasetReader
import torch.optim as optim
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models.crf_tagger import CrfTagger
import wasabi
import pathlib
import time


import parsect.constants as constants
from parsect.utils.common import write_nfold_parscit_train_test

FILES = constants.FILES
PATHS = constants.PATHS
PARSCIT_TRAIN_FILE = FILES["PARSCIT_TRAIN_FILE"]
DATA_DIR = PATHS["DATA_DIR"]


if __name__ == "__main__":
    data_dir = pathlib.Path(DATA_DIR)
    parscit_train_filepath = pathlib.Path(PARSCIT_TRAIN_FILE)
    train_conll_citations_path = data_dir.joinpath("parscit_train_conll.txt")
    test_conll_citations_path = data_dir.joinpath("parscit_test_conll.txt")
    msg_printer = wasabi.Printer()

    for idx, is_write_success in enumerate(
        write_nfold_parscit_train_test(
            parscit_train_filepath=parscit_train_filepath, nsplits=2
        )
    ):
        with msg_printer.loading(f"starting fold {idx}"):
            time.sleep(5)

        if is_write_success:
            reader = Conll2003DatasetReader(
                token_indexers={"tokens": SingleIdTokenIndexer()},
                tag_label="ner",
                label_namespace="labels",
            )
            train_dataset = reader.read(str(train_conll_citations_path))
            test_dataset = reader.read(str(test_conll_citations_path))
            vocab = Vocabulary.from_instances(train_dataset + test_dataset)

            EMBEDDING_DIM = 100
            HIDDEN_DIM = 1024
            token_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size("tokens"),
                embedding_dim=EMBEDDING_DIM,
            )
            word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

            lstm = PytorchSeq2SeqWrapper(
                nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
            )

            model = CrfTagger(
                vocab=vocab,
                text_field_embedder=word_embeddings,
                encoder=lstm,
                label_namespace="labels",
                label_encoding="BIO",
                verbose_metrics=False,
            )

            if torch.cuda.is_available():
                cuda_device = 0
                model = model.cuda(cuda_device)
            else:
                cuda_device = -1
            optimizer = optim.Adam(model.parameters())
            iterator = BasicIterator(batch_size=32)
            iterator.index_with(vocab)
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                iterator=iterator,
                train_dataset=train_dataset,
                validation_dataset=test_dataset,
                patience=10,
                num_epochs=50,
                cuda_device=cuda_device,
            )
            trainer.train()
