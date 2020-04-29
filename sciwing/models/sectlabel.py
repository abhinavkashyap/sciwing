import sciwing.constants as constants
from sciwing.datasets.classification.text_classification_dataset import (
    TextClassificationDatasetManager,
)
from sciwing.modules.embedders.bow_elmo_embedder import BowElmoEmbedder
from sciwing.modules.embedders.word_embedder import WordEmbedder
from sciwing.modules.embedders.concat_embedders import ConcatEmbedders
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from sciwing.models.simpleclassifier import SimpleClassifier
from sciwing.infer.classification.classification_inference import (
    ClassificationInference,
)
from sciwing.api.utils.pdf_reader import PdfReader
from sciwing.utils.common import cached_path, chunks
import pathlib
import json
import wasabi
from typing import List
import itertools

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]


class SectLabel:
    def __init__(self):
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)
        self.final_model_dir = self.models_cache_dir.joinpath("sectlabel_elmo_bilstm")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)
        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()

    def _get_model(self):
        elmo_embedder = BowElmoEmbedder(layer_aggregation="sum")

        # instantiate the vanilla embedder
        vanilla_embedder = WordEmbedder(embedding_type=self.hparams.get("emb_type"))

        # concat the embeddings
        embedder = ConcatEmbedders([vanilla_embedder, elmo_embedder])

        hidden_dim = self.hparams.get("hidden_dim")
        bidirectional = self.hparams.get("bidirectional")
        combine_strategy = self.hparams.get("combine_strategy")

        encoder = LSTM2VecEncoder(
            embedder=embedder,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            combine_strategy=combine_strategy,
        )

        encoding_dim = (
            2 * hidden_dim
            if bidirectional and combine_strategy == "concat"
            else hidden_dim
        )

        model = SimpleClassifier(
            encoder=encoder,
            encoding_dim=encoding_dim,
            num_classes=23,
            classification_layer_bias=True,
            datasets_manager=self.data_manager,
        )

        return model

    def _get_infer_client(self):
        client = ClassificationInference(
            model=self.model,
            model_filepath=self.final_model_dir.joinpath("best_model.pt"),
            datasets_manager=self.data_manager,
        )
        return client

    def predict_for_file(self, filename: str) -> List[str]:
        lines = []
        with open(filename) as fp:
            for line in fp:
                lines.append(line)

        predictions = self.infer.infer_batch(lines=lines)
        for line, prediction in zip(lines, predictions):
            self.msg_printer.text(title=line, text=prediction)

        return predictions

    def predict_for_text(self, text: str) -> str:
        prediction = self.infer.on_user_input(line=text)
        self.msg_printer.text(title=text, text=prediction)
        return prediction

    def predict_for_text_batch(self, texts: List[str]) -> List[str]:
        predictions = self.infer.infer_batch(lines=texts)
        return predictions

    def _get_data(self):
        train_filename = self.data_dir.joinpath("sectLabel.train")
        dev_filename = self.data_dir.joinpath("sectLabel.dev")
        test_filename = self.data_dir.joinpath("sectLabel.test")

        data_manager = TextClassificationDatasetManager(
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
        )

        return data_manager

    def _get_hparams(self):
        with open(self.final_model_dir.joinpath("hyperparams.json")) as fp:
            hyperparams = json.load(fp)
        return hyperparams

    def _download_if_required(self):
        cached_path(
            path=self.final_model_dir,
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/sectlabel_elmo_bilstm.zip",
        )

    def extract_abstract(
        self, pdf_filename: pathlib.Path, dehyphenate: bool = True
    ) -> str:
        """ Extracts abstracts from a pdf using sectlabel. This is the python programmatic version of
        the API. The APIs can be found in sciwing/api. You can see that for more information

        Parameters
        ----------
        pdf_filename : pathlib.Path
            The path where the pdf is stored
        dehyphenate : bool
            Scientific documents are two columns sometimes and there are a lot of hyphenation
            introduced. If this is true, we remove the hyphens from the code

        Returns
        -------
        str
            The abstract of the pdf

        """
        pdf_reader = PdfReader(filepath=pdf_filename)
        lines = pdf_reader.read_pdf()
        all_labels = []
        all_lines = []

        for batch_lines in chunks(lines, 64):
            labels = self.infer.infer_batch(lines=batch_lines)
            all_labels.append(labels)
            all_lines.append(batch_lines)

        all_lines = itertools.chain.from_iterable(all_lines)
        all_lines = list(all_lines)

        all_labels = itertools.chain.from_iterable(all_labels)
        all_labels = list(all_labels)

        response_tuples = []
        for line, label in zip(all_lines, all_labels):
            response_tuples.append((line, label))

        abstract_lines = []
        found_abstract = False
        for line, label in response_tuples:
            if label == "sectionHeader" and line.strip().lower() == "abstract":
                found_abstract = True
                continue
            if found_abstract and label == "sectionHeader":
                break
            if found_abstract:
                abstract_lines.append(line.strip())

        if dehyphenate:
            buffer_lines = []  # holds lines that should be a single line
            final_lines = []
            for line in abstract_lines:
                if line.endswith("-"):
                    line_ = line.replace("-", "")  # replace the hyphen
                    buffer_lines.append(line_)
                else:

                    # if the hyphenation ended on the previous
                    # line then the next line also needs to be
                    # added to the buffer line
                    if len(buffer_lines) > 0:
                        buffer_lines.append(line)

                        line_ = "".join(buffer_lines)

                        # add the line from buffer first
                        final_lines.append(line_)

                    else:
                        # add the current line
                        final_lines.append(line)

                    buffer_lines = []

            abstract_lines = final_lines

        abstract = " ".join(abstract_lines)
        return abstract


if __name__ == "__main__":
    sectlabel = SectLabel()
    abstract = sectlabel.extract_abstract(
        pdf_filename=pathlib.Path(
            "/Users/abhinav/Downloads/transferring_knowledge_from_discourse_to_arguments.pdf"
        ),
        dehyphenate=True,
    )
    abstract = abstract.encode("utf-8").decode("utf-8")
    print(f"abstract\n =============== \n {abstract}")
