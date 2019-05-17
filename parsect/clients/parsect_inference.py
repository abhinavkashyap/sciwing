import json
from parsect.datasets.parsect_dataset import ParsectDataset
import parsect.constants as constants
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

FILES = constants.FILES

SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']


class ParsectInference:
    """
    The parsect engine runs the test lines through the classifier
    and returns the predictions/probabilities for different classes
    At a later point in time this method should be able to take any
    context of lines (may be from a file) and produce the output.

    This class also helps in performing various interactions with
    the results on the test dataset.
    Some features are
    1) Show confusion matrix
    2) Investigate a particular example in the test dataset
    3) Get instances that were classified as 2 when their true label is 1 and others
    """
    def __init__(self,
                 model: torch.nn.Module,
                 model_filepath: str,
                 hyperparam_config_filepath: str):
        """
        :param model: type: torch.nn.Module
        Pass the model on which inference should be run
        :param model_filepath: type: str
        The model filepath is the chkpoint file where the model state is stored
        :param hyperparam_config_filepath: type: str
        The path where all hyper-parameters necessary for restoring the model
        is necessary
        """
        self.model = model
        self.model_filepath = model_filepath
        self.hyperparam_config_filename = hyperparam_config_filepath

        with open(self.hyperparam_config_filename, 'r') as fp:
            config = json.load(fp)

        self.max_num_words = config['MAX_NUM_WORDS']
        self.max_length = config['MAX_LENGTH']
        self.vocab_store_location = config['VOCAB_STORE_LOCATION']
        self.debug = config['DEBUG']
        self.debug_dataset_proportion = config['DEBUG_DATASET_PROPORTION']
        self.batch_size = config['BATCH_SIZE']
        self.emb_dim = config['EMBEDDING_DIMENSION']
        self.lr = config['LEARNING_RATE']
        self.num_epochs = config['NUM_EPOCHS']
        self.save_every = config['SAVE_EVERY']

        self.test_dataset = self.get_test_dataset()
        self.load_model()
        self.run_inference()

    def get_test_dataset(self) -> Dataset:
        test_dataset = ParsectDataset(
            secthead_label_file=SECT_LABEL_FILE,
            dataset_type='test',
            max_num_words=self.max_num_words,
            max_length=self.max_length,
            vocab_store_location=self.vocab_store_location,
            debug=self.debug,
            debug_dataset_proportion=self.debug_dataset_proportion
        )
        return test_dataset

    def load_model(self):

        model_chkpoint = torch.load(self.model_filepath)
        model_state_dict = model_chkpoint['model_state']
        self.model.load_state_dict(model_state_dict)

    def run_inference(self):
        loader = DataLoader(dataset=self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False)
        pass 

