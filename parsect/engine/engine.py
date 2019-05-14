from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from wasabi import Printer
import multiprocessing
from typing import Tuple, Iterator


class Engine:
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset,
                 optimizer:optim,
                 batch_size: int,
                 save_dir: str,
                 num_epochs: int):
        """
        This orchestrates the whole model training. The supervised machine learning
        that needs to be used
        :param model: type: nnn.Module
        Any pytorch model that takes in inputs
        :param train_dataset: type: torch.utils.data.Dataset
        training dataset, that can be iterated and follows the Dataset conventions of PyTorch
        :param validation_dataset: type: torch.utils.data.Dataset
        Validation dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param test_dataset: type: torch.utils.Dataset
        Test Dataset, that can be iterated and follows the Dataset convention of PyTorch
        :param optimizer: torch.optim
        Any optimizer that belongs to the torch.optim module
        :param batch_size: type: int
        Batch size for the train, validation and test dataset
        :param save_dir: type: str
        The full location where the intermediate results are stored
        :param num_epochs: type: int
        Number of epochs to run traininng
        """

        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.msg_printer = Printer()

        self.num_workers = multiprocessing.cpu_count()  # num_workers

        # get the data loader
        # TODO: For now we randomly sample the dataset to obtain instances
        self.train_loader = self.get_loader(self.train_dataset)
        self.validation_loader = self.get_loader(self.validation_dataset)
        self.test_loader = self.get_loader(self.test_dataset)

        # refresh the iters at the beginning of every epoch
        self.train_iter = None
        self.validation_iter = None
        self.test_iter = None

    def get_loader(self, dataset: Dataset) -> DataLoader:
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        return loader

    def run(self):
        """
        Run the engine
        :return:
        """
        for epoch_num in range(self.num_epochs):
            self.train_epoch()

            self.validation_epoch()

    def train_epoch(self):
        """
        Run the training for one epoch
        """
        train_iter = self.get_iter(self.train_loader)
        self.model.train()
        try:
            # N*T, N * 1, N * 1
            tokens, labels, len_tokens = next(train_iter)
            labels = labels.squeeze(1)
            model_forward_out = self.model(tokens, labels, is_training=True)

            try:
                self.optimizer.zero_grad()
                loss = model_forward_out['loss']
                loss.backward()
                self.optimizer.step()

            except KeyError:
                self.msg_printer.fail('The model output dictionary does not have '
                                      'a key called loss. Please check to have '
                                      'loss in the model output')
        except StopIteration:
            pass

    def validation_epoch(self):
        """
        Run the validation
        """
        pass

    def get_train_dataset(self):
        return self.train_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset

    @staticmethod
    def get_iter(loader: DataLoader) -> Iterator:
        """
        The iterators return the next batch of instances
        when they are called. This will be useful
        for going over the dataset in a batch wise manner
        :return:
        """
        iterator = iter(loader)
        return iterator


if __name__ == '__main__':
    import os
    import parsect.constants as constants
    from parsect.datasets.parsect_dataset import ParsectDataset
    from parsect.modules.bow_encoder import BOW_Encoder
    from parsect.models.simple_classifier import Simple_Classifier
    from torch.nn import Embedding
    import numpy as np
    FILES = constants.FILES
    SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']

    MAX_NUM_WORDS = 1000
    MAX_LENGTH = 50
    vocab_store_location = os.path.join('.', 'vocab.json')
    DEBUG = True

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='train',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='valid',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='test',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG
    )

    BATCH_SIZE = 1
    NUM_TOKENS = 3
    EMB_DIM = 300
    VOCAB_SIZE = 10
    NUM_CLASSES = 3
    embedding = Embedding.from_pretrained(torch.zeros([VOCAB_SIZE, EMB_DIM]))
    labels = torch.LongTensor([1])

    encoder = BOW_Encoder(emb_dim=EMB_DIM,
                          embedding=embedding,
                          dropout_value=0,
                          aggregation_type='sum')
    tokens = np.random.randint(0, VOCAB_SIZE - 1, size=(BATCH_SIZE, NUM_TOKENS))
    tokens = torch.LongTensor(tokens)
    model = Simple_Classifier(encoder=encoder,
                              encoding_dim=EMB_DIM,
                              num_classes=NUM_CLASSES,
                              classification_layer_bias=False
                              )

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    engine = Engine(model,
                    train_dataset,
                    validation_dataset,
                    test_dataset,
                    optimizer=optimizer,
                    batch_size=BATCH_SIZE,
                    save_dir=os.path.join('.'),
                    num_epochs=1)
