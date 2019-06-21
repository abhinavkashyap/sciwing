import parsect.constants as constants
import json
from parsect.models.bert_seq_classifier import BertSeqClassifier
from parsect.clients.parsect_inference import ParsectInference
from parsect.datasets.parsect_dataset import ParsectDataset
from parsect.tokenizers.bert_tokenizer import TokenizerForBert
import os

PATHS = constants.PATHS
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
FILES = constants.FILES
SECT_LABEL_FILE = FILES["SECT_LABEL_FILE"]


def get_bert_seq_classifier_infer(dirname: str):
    hyperparam_config_filepath = os.path.join(dirname, "config.json")
    with open(hyperparam_config_filepath, "r") as fp:
        config = json.load(fp)

    EMBEDDING_DIM = config["EMBEDDING_DIMENSION"]
    NUM_CLASSES = config["NUM_CLASSES"]
    MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
    MAX_NUM_WORDS = config["MAX_NUM_WORDS"]
    MAX_LENGTH = config["MAX_LENGTH"]
    VOCAB_STORE_LOCATION = config["VOCAB_STORE_LOCATION"]
    DEBUG = config["DEBUG"]
    DEBUG_DATASET_PROPORTION = config["DEBUG_DATASET_PROPORTION"]
    BERT_TYPE = config["BERT_TYPE"]
    EMBEDDING_TYPE = config.get("EMBEDDING_TYPE", None)

    model_filepath = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

    model = BertSeqClassifier(
        num_classes=NUM_CLASSES,
        emb_dim=EMBEDDING_DIM,
        bert_type=BERT_TYPE,
        dropout_value=0.0,
    )

    bert_tokenizer = TokenizerForBert(bert_type=BERT_TYPE)
    dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type="test",
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION,
        embedding_type=EMBEDDING_TYPE,
        embedding_dimension=EMBEDDING_DIM,
        tokenizer=bert_tokenizer,
        tokenization_type="bert",
        start_token="[CLS]",
        end_token="[SEP]",
        pad_token="[PAD]",
    )

    inference = ParsectInference(
        model=model,
        model_filepath=model_filepath,
        hyperparam_config_filepath=hyperparam_config_filepath,
        dataset=dataset,
    )

    return inference


if __name__ == "__main__":
    dirname = os.path.join(
        OUTPUT_DIR, "debug_bert_seq_classifier_base_cased_emb_lc_10e_1e-2lr"
    )
    infer = get_bert_seq_classifier_infer(dirname)
