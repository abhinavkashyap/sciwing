import pytest
from parsect.infer.sci_ie_inference import ScienceIEInference
from parsect.datasets.seq_labeling.science_ie_dataset import ScienceIEDataset
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.charlstm_encoder import CharLSTMEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.embedders.concat_embedders import ConcatEmbedders
from parsect.models.science_ie_tagger import ScienceIETagger
from parsect.utils.science_ie_data_utils import ScienceIEDataUtils
import parsect.constants as constants
import pathlib
import json
import torch.nn as nn
import torch

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
FILES = constants.FILES
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]


@pytest.fixture
def setup_science_ie_inference():
    debug_parscit_model_folder = pathlib.Path(OUTPUT_DIR, "lstm_crf_scienceie_debug")
    hyperparam_config_filename = debug_parscit_model_folder.joinpath("config.json")
    sci_ie_dev_utils = ScienceIEDataUtils(
        folderpath=pathlib.Path(SCIENCE_IE_DEV_FOLDER), ignore_warnings=True
    )
    sci_ie_dev_utils.write_bilou_lines(
        out_filename=pathlib.Path(DATA_DIR, "dev_science_ie.txt")
    )
    sci_ie_dev_utils.merge_files(
        task_filename=pathlib.Path(DATA_DIR, "dev_science_ie_task_conll.txt"),
        process_filename=pathlib.Path(DATA_DIR, "dev_science_ie_process_conll.txt"),
        material_filename=pathlib.Path(DATA_DIR, "dev_science_ie_material_conll.txt"),
        out_filename=pathlib.Path(DATA_DIR, "dev_science_ie.txt"),
    )

    with open(hyperparam_config_filename, "r") as fp:
        config = json.load(fp)

    MAX_NUM_WORDS = config.get("MAX_NUM_WORDS", None)
    MAX_LENGTH = config.get("MAX_LENGTH", None)
    MAX_CHAR_LENGTH = config.get("MAX_CHAR_LENGTH", None)
    VOCAB_STORE_LOCATION = config.get("VOCAB_STORE_LOCATION", None)
    DEBUG = config.get("DEBUG", None)
    DEBUG_DATASET_PROPORTION = config.get("DEBUG_DATASET_PROPORTION", None)
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", None)
    EMBEDDING_DIMENSION = config.get("EMBEDDING_DIMENSION", None)
    HIDDEN_DIMENSION = config.get("HIDDEN_DIM", None)
    BIDIRECTIONAL = config.get("BIDIRECTIONAL", None)
    COMBINE_STRATEGY = config.get("COMBINE_STRATEGY", None)
    DEVICE = config.get("DEVICE", "cpu")
    NUM_CLASSES = config.get("NUM_CLASSES", None)
    MODEL_SAVE_DIR = config.get("MODEL_SAVE_DIR", None)
    model_filepath = pathlib.Path(MODEL_SAVE_DIR, "best_model.pt")
    CHAR_VOCAB_STORE_LOCATION = config.get("CHAR_VOCAB_STORE_LOCATION", None)
    CHAR_EMBEDDING_DIMENSION = config.get("CHAR_EMBEDDING_DIMENSION", None)
    USE_CHAR_ENCODER = config.get("USE_CHAR_ENCODER", None)
    CHAR_ENCODER_HIDDEN_DIM = config.get("CHAR_ENCODER_HIDDEN_DIM", None)

    test_science_ie_conll_filepath = pathlib.Path(DATA_DIR, "dev_science_ie.txt")

    test_dataset = ScienceIEDataset(
        science_ie_conll_file=test_science_ie_conll_filepath,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_word_length=MAX_LENGTH,
        max_char_length=MAX_CHAR_LENGTH,
        word_vocab_store_location=VOCAB_STORE_LOCATION,
        char_vocab_store_location=CHAR_VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        word_embedding_type=EMBEDDING_TYPE,
        word_embedding_dimension=EMBEDDING_DIMENSION,
        character_embedding_dimension=CHAR_EMBEDDING_DIMENSION,
        start_token="<SOS>",
        end_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        word_add_start_end_token=False,
    )

    embedding = test_dataset.get_preloaded_word_embedding()
    embedding = nn.Embedding.from_pretrained(embedding)

    char_embedding = test_dataset.get_preloaded_char_embedding()
    char_embedding = nn.Embedding.from_pretrained(char_embedding)

    embedder = VanillaEmbedder(embedding_dim=EMBEDDING_DIMENSION, embedding=embedding)

    if USE_CHAR_ENCODER:
        char_embedder = VanillaEmbedder(
            embedding_dim=CHAR_EMBEDDING_DIMENSION, embedding=char_embedding
        )
        char_encoder = CharLSTMEncoder(
            char_emb_dim=CHAR_EMBEDDING_DIMENSION,
            char_embedder=char_embedder,
            bidirectional=True,
            hidden_dim=CHAR_ENCODER_HIDDEN_DIM,
            combine_strategy="concat",
        )
        embedder = ConcatEmbedders([embedder, char_encoder])
        EMBEDDING_DIMENSION += 2 * CHAR_ENCODER_HIDDEN_DIM

    lstm2seqencoder = Lstm2SeqEncoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedder=embedder,
        dropout_value=0.0,
        hidden_dim=HIDDEN_DIMENSION,
        bidirectional=BIDIRECTIONAL,
        combine_strategy=COMBINE_STRATEGY,
        rnn_bias=True,
        device=torch.device(DEVICE),
    )
    model = ScienceIETagger(
        rnn2seqencoder=lstm2seqencoder,
        num_classes=NUM_CLASSES,
        hid_dim=2 * HIDDEN_DIMENSION
        if BIDIRECTIONAL and COMBINE_STRATEGY == "concat"
        else HIDDEN_DIMENSION,
    )

    inference_client = ScienceIEInference(
        model=model,
        model_filepath=str(model_filepath),
        hyperparam_config_filepath=str(hyperparam_config_filename),
        dataset=test_dataset,
    )
    return inference_client


class TestScienceIEInference:
    def test_print_prf_table_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        try:
            inference.print_prf_table()
        except:
            pytest.fail("Print PRF Table fails")

    def test_print_confusion_matrix_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        try:
            inference.print_confusion_matrix()
        except:
            pytest.fail("Print Confusion Matrix fails")

    @pytest.mark.parametrize("first_class", range(0, 8))
    @pytest.mark.parametrize("second_class", range(0, 8))
    def test_get_misclassified_sentences_works(
        self, setup_science_ie_inference, first_class, second_class
    ):
        inference = setup_science_ie_inference
        try:
            inference.get_misclassified_sentences(first_class, second_class)
        except:
            pytest.fail("Getting Misclassified Sentences works")

    def test_generate_report_for_paper_works(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        try:
            inference.generate_report_for_paper()
        except:
            pytest.fail("Generate Report For paper does not work")

    def test_science_ie_user_input(self, setup_science_ie_inference):
        inference = setup_science_ie_inference
        try:
            inference.infer_single_sentence("This is a sentence")
        except:
            pytest.fail("On user input does not work")

    def test_scienceie_generate_pred_folder(self, setup_science_ie_inference, tmpdir):
        inference = setup_science_ie_inference
        try:
            dev_folder = pathlib.Path(SCIENCE_IE_DEV_FOLDER)
            pred_folder = tmpdir.mkdir("science_ie_pred_folder")
            pred_folder = pathlib.Path(pred_folder)
            inference.generate_predict_folder(
                dev_folder=dev_folder, pred_folder=pred_folder
            )
        except:
            pytest.fail("Generating Prediction Folder does not work")
