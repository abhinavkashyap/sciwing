from sciwing.utils.vis_seq_tags import VisTagging
from sciwing.modules.embedders.trainable_word_embedder import TrainableWordEmbedder
from sciwing.modules.embedders.char_embedder import CharEmbedder
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2seqencoder import Lstm2SeqEncoder
from sciwing.models.rnn_seq_crf_tagger import RnnSeqCrfTagger
from sciwing.datasets.seq_labeling.seq_labelling_dataset import (
    SeqLabellingDatasetManager,
)
from sciwing.infer.seq_label_inference.seq_label_inference import (
    SequenceLabellingInference,
)
from sciwing.cli.sciwing_interact import SciWINGInteract
from sciwing.utils.common import cached_path
import sciwing.constants as constants
import pathlib
import json
import torch.nn as nn
import wasabi
from typing import List
from collections import defaultdict
import torch
from typing import Optional, Tuple

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
DATA_FILE_URLS = constants.DATA_FILE_URLS


class NeuralParscit(nn.Module):
    """ It defines a neural parscit model. The model is used for citation string parsing. This model
    helps you use a pre-trained model who architecture is fixed and is trained by SciWING.
    You can also fine-tune the model on your own dataset.

    For practitioners, we provide ways to obtain results quickly from a set of citations
    stored in a file or from a string. If you want to see the demo head over to our demo site.

    """

    def __init__(self, device: Optional[Tuple[torch.device, int]] = -1):
        super(NeuralParscit, self).__init__()

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, int):
            if device == -1:
                device_string = "cpu"
            else:
                device_string = f"cuda:{device}"
            self.device = torch.device(device_string)
        else:
            raise ValueError(
                f"Pass the device number or the device object from Pytorch"
            )

        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("lstm_crf_parscit_final")
        if not self.models_cache_dir.is_dir():
            self.models_cache_dir.mkdir(parents=True)
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)

        if not self.data_dir.is_dir():
            self.data_dir.mkdir(parents=True)

        self.train_data_file_url = DATA_FILE_URLS["PARSCIT_TRAIN"]
        self.dev_data_file_url = DATA_FILE_URLS["PARSCIT_DEV"]
        self.test_data_file_url = DATA_FILE_URLS["PARSCIT_TEST"]
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.hparams = self._get_hparams()
        self.data_manager = self._get_data()
        self.model: nn.Module = self._get_model()
        self.infer = self._get_infer_client()
        self.vis_tagger = VisTagging()
        self.interact_ = SciWINGInteract(self.infer)

    def _get_model(self) -> nn.Module:
        word_embedder = TrainableWordEmbedder(
            embedding_type=self.hparams.get("emb_type"),
            datasets_manager=self.data_manager,
            device=self.device,
        )

        char_embedder = CharEmbedder(
            char_embedding_dimension=self.hparams.get("char_emb_dim"),
            hidden_dimension=self.hparams.get("char_encoder_hidden_dim"),
            datasets_manager=self.data_manager,
            device=self.device,
        )

        elmo_embedder = BowElmoEmbedder(
            datasets_manager=self.data_manager,
            layer_aggregation="sum",
            device=self.device,
        )

        embedder = ConcatEmbedders([word_embedder, char_embedder, elmo_embedder])

        lstm2seqencoder = Lstm2SeqEncoder(
            embedder=embedder,
            hidden_dim=self.hparams.get("hidden_dim"),
            bidirectional=self.hparams.get("bidirectional"),
            combine_strategy=self.hparams.get("combine_strategy"),
            rnn_bias=True,
            dropout_value=self.hparams.get("lstm2seq_dropout", 0.0),
            add_projection_layer=False,
            device=self.device,
        )
        model = RnnSeqCrfTagger(
            rnn2seqencoder=lstm2seqencoder,
            encoding_dim=2 * self.hparams.get("hidden_dim")
            if self.hparams.get("bidirectional")
            and self.hparams.get("combine_strategy") == "concat"
            else self.hparams.get("hidden_dim"),
            datasets_manager=self.data_manager,
            device=self.device,
        )

        return model

    def _get_infer_client(self):
        infer_client = SequenceLabellingInference(
            model=self.model,
            model_filepath=self.final_model_dir.joinpath("best_model.pt"),
            datasets_manager=self.data_manager,
            device=self.device,
        )
        return infer_client

    def _predict(self, line: str):
        predictions = self.infer.on_user_input(line=line)
        return predictions

    def predict_for_file(self, filename: str) -> List[str]:
        """ Parse the references in a file where every line is a reference

        Parameters
        ----------
        filename : str
            The filename where the references are stored

        Returns
        -------
        List[str]
            A list of parsed tags

        """
        predictions = defaultdict(list)
        with open(filename, "r") as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                pred_ = self._predict(line=line)
                for namespace, prediction in pred_.items():
                    predictions[namespace].append(prediction[0])
                    stylized_string = self.vis_tagger.visualize_tokens(
                        text=line.split(), labels=prediction[0].split()
                    )
                    self.msg_printer.divider(
                        f"Predictions for Line: {line_idx+1} from {filename}"
                    )
                    print(stylized_string)
                    print("\n")

        return predictions[self.data_manager.label_namespaces[0]]

    def predict_for_text(self, text: str, show=True) -> str:
        """ Parse the citation string for the given text

        Parameters
        ----------
        text : str
            reference string to parse
        show : bool
            If `True`, then we print the stylized string - where the stylized string provides
            different colors for different tags
            If `False` - then we do not print the stylized string

        Returns
        -------
        str
            The parsed citation string

        """
        predictions = self._predict(line=text)
        for namespace, prediction in predictions.items():
            if show:
                self.msg_printer.divider(f"Prediction for {namespace.upper()}")
                stylized_string = self.vis_tagger.visualize_tokens(
                    text=text.split(), labels=prediction[0].split()
                )
                print(stylized_string)
            return prediction[0]

    def _get_data(self):
        data_manager = SeqLabellingDatasetManager(
            train_filename=cached_path(
                path=self.data_dir.joinpath("parscit.train"),
                url=self.train_data_file_url,
                unzip=False,
            ),
            dev_filename=cached_path(
                path=self.data_dir.joinpath("parscit.dev"),
                url=self.dev_data_file_url,
                unzip=False,
            ),
            test_filename=cached_path(
                path=self.data_dir.joinpath("parscit.test"),
                url=self.test_data_file_url,
                unzip=False,
            ),
        )
        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        # download the model weights and data to client machine
        cached_path(
            path=f"{self.final_model_dir}.zip",
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/lstm_crf_parscit_final.zip",
            unzip=True,
        )

    def interact(self):
        """ Interact with the pretrained model
        You can also interact from command line using `sciwing interact neural-parscit`
        """
        self.interact_.interact()


if __name__ == "__main__":
    neural_parscit = NeuralParscit(device=0)
