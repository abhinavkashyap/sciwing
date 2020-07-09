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
from sciwing.cli.sciwing_interact import SciWINGInteract
import pathlib
import json
import wasabi
from typing import List
import itertools
from logzero import setup_logger
import logging
from tqdm import tqdm
from sciwing.utils.common import cached_path

PATHS = constants.PATHS
MODELS_CACHE_DIR = PATHS["MODELS_CACHE_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
DATA_FILE_URLS = constants.DATA_FILE_URLS


class SectLabel:
    def __init__(self, log_file: str = None, device: str = "cpu"):
        self.device = device
        self.models_cache_dir = pathlib.Path(MODELS_CACHE_DIR)

        if not self.models_cache_dir.is_dir():
            self.models_cache_dir.mkdir(parents=True)

        self.final_model_dir = self.models_cache_dir.joinpath("sectlabel_elmo_bilstm")
        self.model_filepath = self.final_model_dir.joinpath("best_model.pt")
        self.data_dir = pathlib.Path(DATA_DIR)

        if not self.data_dir.is_dir():
            self.data_dir.mkdir(parents=True)

        self.train_data_url = DATA_FILE_URLS["SECT_LABEL_TRAIN_FILE"]
        self.dev_data_url = DATA_FILE_URLS["SECT_LABEL_DEV_FILE"]
        self.test_data_url = DATA_FILE_URLS["SECT_LABEL_TEST_FILE"]

        self.msg_printer = wasabi.Printer()
        self._download_if_required()
        self.data_manager = self._get_data()
        self.hparams = self._get_hparams()
        self.model = self._get_model()
        self.infer = self._get_infer_client()
        self.cli_interact = SciWINGInteract(self.infer)
        self.log_file = log_file

        if log_file:
            self.logger = setup_logger(
                "sectlabel_logger", logfile=self.log_file, level=logging.INFO
            )
        else:
            self.logger = self.msg_printer

    def _get_model(self):
        elmo_embedder = BowElmoEmbedder(layer_aggregation="sum", device=self.device)

        # instantiate the vanilla embedder
        vanilla_embedder = WordEmbedder(
            embedding_type=self.hparams.get("emb_type"), device=self.device
        )

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
            device=self.device,
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
            device=self.device,
        )
        model.to(self.device)

        return model

    def _get_infer_client(self):
        client = ClassificationInference(
            model=self.model,
            model_filepath=self.final_model_dir.joinpath("best_model.pt"),
            datasets_manager=self.data_manager,
            device=self.device,
        )
        return client

    def predict_for_file(self, filename: str) -> List[str]:
        """ Predicts the logical sections for all the sentences in a file, with one sentence per line

        Parameters
        ----------
        filename : str
            The path of the file

        Returns
        -------
        List[str]
            The predictions for each line.

        """
        lines = []
        with open(filename) as fp:
            for line in fp:
                lines.append(line)

        predictions = self.infer.infer_batch(lines=lines)
        for line, prediction in zip(lines, predictions):
            self.msg_printer.text(title=line, text=prediction)

        return predictions

    def predict_for_pdf(self, pdf_filename: pathlib.Path) -> (List[str], List[str]):
        """ Predicts lines and labels given a pdf filename

        Parameters
        ----------
        pdf_filename : pathlib.Path
            The location where pdf files are stored

        Returns
        -------
        List[str], List[str]
            The lines and labels inferred on the file
        """
        pdf_reader = PdfReader(filepath=pdf_filename)
        lines = pdf_reader.read_pdf()

        lines = self._preprocess(lines)

        if len(lines) == 0:
            self.logger.warning(f"No lines were read from file {pdf_filename}")
            return ""

        all_labels = []
        all_lines = []

        for batch_lines in chunks(lines, 64):
            labels = self.infer.infer_batch(lines=batch_lines)
            all_labels.append(labels)
            all_lines.append(batch_lines)

        all_lines = itertools.chain.from_iterable(all_lines)
        all_labels = itertools.chain.from_iterable(all_labels)
        all_lines = list(all_lines)
        all_labels = list(all_labels)

        return all_lines, all_labels

    def predict_for_text(self, text: str) -> str:
        """ Predicts the logical section that the line belongs to

        Parameters
        ----------
        text: str
            A single line of text

        Returns
        -------
        str
            The logical section of the text.

        """
        prediction = self.infer.on_user_input(line=text)
        self.msg_printer.text(title=text, text=prediction)
        return prediction

    def predict_for_text_batch(self, texts: List[str]) -> List[str]:
        """ Predicts the logical section for a batch of text.

        Parameters
        ----------
        texts: List[str]
            A batch of text

        Returns
        -------
        List[str]
            A batch of predictions

        """
        predictions = self.infer.infer_batch(lines=texts)
        return predictions

    def _get_data(self):
        train_filename = self.data_dir.joinpath("sectLabel.train")
        dev_filename = self.data_dir.joinpath("sectLabel.dev")
        test_filename = self.data_dir.joinpath("sectLabel.test")

        train_filename = cached_path(
            path=train_filename, url=self.train_data_url, unzip=False
        )
        dev_filename = cached_path(
            path=dev_filename, url=self.dev_data_url, unzip=False
        )

        test_filename = cached_path(
            path=test_filename, url=self.test_data_url, unzip=False
        )

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
            path=f"{self.final_model_dir}.zip",
            url="https://parsect-models.s3-ap-southeast-1.amazonaws.com/sectlabel_elmo_bilstm.zip",
            unzip=True,
        )

    @staticmethod
    def _preprocess(lines: str):
        preprocessed_lines = []
        for line in lines:
            line_ = line.strip()
            if bool(line_):
                line_words = line_.split()
                num_single_character_words = sum(
                    [1 for word in line_words if len(word) == 1]
                )
                num_words = len(line_words)
                percentage_single_character_words = (
                    num_single_character_words / num_words
                ) * 100
                if percentage_single_character_words > 40:
                    line_ = "".join(line_words)
                    preprocessed_lines.append(line_)
                else:
                    preprocessed_lines.append(line_)
        return preprocessed_lines

    @staticmethod
    def _extract_abstract_for_file(lines: List[str], labels: List[str]) -> List[str]:
        """ Given the linse

        Parameters
        ----------
        lines: List[str]
            A set of lines
        labels: List[str]
            A set of labels

        Returns
        -------
        List[str]
            Lines in the abstract

        """
        response_tuples = []
        for line, label in zip(lines, labels):
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

        return abstract_lines

    def dehyphenate(self, lines: List[str]) -> List[str]:
        """ Dehyphenates a list of strings

        Parameters
        ----------
        lines: List[str]
            A list of hyphenated strings

        Returns
        -------
        List[str]
            A list of dehyphenated strings
        """
        buffer_lines = []  # holds lines that should be a single line
        final_lines = []
        for line in lines:
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
        return final_lines

    def extract_abstract_for_file(
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
        self.msg_printer.info(f"Extracting abstract for {pdf_filename}")
        all_lines, all_labels = self.predict_for_pdf(pdf_filename=pdf_filename)
        abstract_lines = self._extract_abstract_for_file(
            lines=all_lines, labels=all_labels
        )

        if dehyphenate:
            abstract_lines = self.dehyphenate(abstract_lines)

        abstract = " ".join(abstract_lines)
        return abstract

    def extract_abstract_for_folder(self, foldername: pathlib.Path, dehyphenate=True):
        """ Extracts the abstracts for all the pdf fils stored in a folder

        Parameters
        ----------
        foldername : pathlib.Path
            THe path of the folder containing pdf files
        dehyphenate : bool
            We will try to dehyphenate the lines. Useful if the pdfs are two column research paper

        Returns
        -------
        None
            Writes the abstracts to files

        """
        num_files = sum([1 for file in foldername.iterdir()])
        for file in tqdm(
            foldername.iterdir(), total=num_files, desc="Extracting Abstracts"
        ):
            if file.suffix == ".pdf":
                abstract = self.extract_abstract_for_file(
                    pdf_filename=file, dehyphenate=dehyphenate
                )
                self.msg_printer.text(title="abstract", text=abstract)
                with open(f"{file.stem}.abstract", "w") as fp:
                    fp.write(abstract)
                    fp.write("\n")

    @staticmethod
    def _extract_section_headers(lines: List[str], labels: List[str]) -> List[str]:
        section_headers = []
        for line, label in zip(lines, labels):
            if label == "sectionHeader" or label == "subsectionHeader":
                section_headers.append(line.strip())

        return section_headers

    def _extract_references(self, lines: List[str], labels: List[str]) -> List[str]:
        references = []
        for line, label in zip(lines, labels):
            if label == "reference":
                references.append(line.strip())

        # references = self.dehyphenate(references)

        return references

    def extract_all_info(self, pdf_filename: pathlib.Path):
        """ Extracts information from the pdf file.

        Parameters
        ----------
        pdf_filename: pathlib.Path
            The path of the pdf file

        Returns
        -------
        Dict[str, Any]
            A dictionary containing information parsed from the pdf file

        """
        all_lines, all_labels = self.predict_for_pdf(pdf_filename=pdf_filename)
        abstract = self._extract_abstract_for_file(lines=all_lines, labels=all_labels)
        abstract = " ".join(abstract)
        section_headers = self._extract_section_headers(
            lines=all_lines, labels=all_labels
        )
        reference_strings = self._extract_references(lines=all_lines, labels=all_labels)

        return {
            "abstract": abstract,
            "section_headers": section_headers,
            "references": reference_strings,
        }

    def interact(self):
        """ Interact with the pre-trained model
        """
        self.cli_interact.interact()


if __name__ == "__main__":
    sect_label = SectLabel()
    sect_label.interact()
