import pathlib
import parsect.constants as constants
from parsect.utils.science_ie_data_utils import ScienceIEDataUtils
from parsect.datasets.seq_labeling.science_ie_dataset import ScienceIEDataset
from parsect.models.science_ie_tagger import ScienceIETagger
from parsect.modules.lstm2seqencoder import Lstm2SeqEncoder
from parsect.modules.embedders.vanilla_embedder import VanillaEmbedder
from parsect.modules.embedders.concat_embedders import ConcatEmbedders
from parsect.modules.charlstm_encoder import CharLSTMEncoder
import json
import torch
import torch.nn as nn
from parsect.infer.sci_ie_inference import ScienceIEInference

FILES = constants.FILES
PATHS = constants.PATHS
SCIENCE_IE_DEV_FOLDER = FILES["SCIENCE_IE_DEV_FOLDER"]
DATA_DIR = PATHS["DATA_DIR"]


def get_science_ie_infer(dirname: str):
    model_folder = pathlib.Path(dirname)
    hyperparam_config_filename = model_folder.joinpath("config.json")
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
    NUM_LAYERS = config.get("NUM_LAYERS", 1)

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

    embedder = VanillaEmbedder(embedding=embedding, embedding_dim=EMBEDDING_DIMENSION)

    if USE_CHAR_ENCODER:
        char_embedder = VanillaEmbedder(
            embedding=char_embedding, embedding_dim=CHAR_EMBEDDING_DIMENSION
        )
        char_encoder = CharLSTMEncoder(
            char_embedder=char_embedder,
            char_emb_dim=CHAR_EMBEDDING_DIMENSION,
            bidirectional=True,
            hidden_dim=CHAR_ENCODER_HIDDEN_DIM,
            combine_strategy="concat",
            device=torch.device(DEVICE),
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
        NUM_LAYERS=NUM_LAYERS,
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
