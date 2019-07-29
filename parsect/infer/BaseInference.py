from abc import ABCMeta, abstractmethod
import torch.nn as nn
import json
import torch
import wasabi
from typing import Dict, Any


class BaseInference(metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        model_filepath: str,
        hyperparam_config_filepath: str,
        dataset,
    ):
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
        self.test_dataset = dataset

        with open(self.hyperparam_config_filename, "r") as fp:
            config = json.load(fp)

        self.max_num_words = config.get("MAX_NUM_WORDS", None)
        self.max_length = config.get("MAX_LENGTH", None)
        self.vocab_store_location = config.get("VOCAB_STORE_LOCATION", None)
        self.char_vocab_store_location = config.get("CHAR_VOCAB_STORE_LOCATION", None)
        self.max_char_length = config.get("MAX_CHAR_LENGTH", None)
        self.debug = config.get("DEBUG", None)
        self.debug_dataset_proportion = config.get("DEBUG_DATASET_PROPORTION", None)
        self.batch_size = config.get("BATCH_SIZE", None)
        self.emb_dim = config.get("EMBEDDING_DIMENSION", None)
        self.lr = config.get("LEARNING_RATE", None)
        self.num_epochs = config.get("NUM_EPOCHS", None)
        self.save_every = config.get("SAVE_EVERY", None)
        self.model_save_dir = config.get("MODEL_SAVE_DIR", None)
        self.vocab_size = config.get("VOCAB_SIZE", None)
        self.num_classes = config.get("NUM_CLASSES", None)
        self.embedding_type = config.get("EMBEDDING_TYPE", None)
        self.embedding_dimension = config.get("EMBEDDING_DIMENSION", None)
        self.return_instances = config.get("RETURN_INSTANCES", None)
        self.device = torch.device(config.get("DEVICE", "cpu"))
        self.msg_printer = wasabi.Printer()

    def load_model(self):

        with self.msg_printer.loading(
            "LOADING MODEL FROM FILE {0}".format(self.model_filepath)
        ):
            model_chkpoint = torch.load(self.model_filepath)
            model_state_dict = model_chkpoint["model_state"]
            loss_value = model_chkpoint["loss"]
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

        self.msg_printer.good(
            "Loaded Best Model with loss value {0}".format(loss_value)
        )

    @abstractmethod
    def run_inference(self) -> Dict[str, Any]:
        pass
