from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.parsect_dataset import ParsectDataset
from parsect.modules.bow_encoder import BOW_Encoder
import parsect.constants as constants
import os
import torch.nn as nn
import torch.optim as optim
from parsect.engine.engine import Engine

FILES = constants.FILES
PATHS = constants.PATHS

SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']
OUTPUT_DIR = PATHS['OUTPUT_DIR']

if __name__ == '__main__':
    EXP_NAME = 'test_run_1'
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, 'checkpoints')
    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    MAX_NUM_WORDS = 3000
    MAX_LENGTH = 15
    vocab_store_location = os.path.join(EXP_DIR_PATH, 'vocab.json')
    DEBUG = True
    DEBUG_DATASET_PROPORTION = 0.01
    BATCH_SIZE = 10
    EMBEDDING_DIMENSION = 300
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 20
    SAVE_EVERY = 1

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='train',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='valid',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='test',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=vocab_store_location,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION
    )

    VOCAB_SIZE = train_dataset.vocab.get_vocab_len()
    NUM_CLASSES = train_dataset.get_num_classes()
    embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
    encoder = BOW_Encoder(
        emb_dim=EMBEDDING_DIMENSION,
        embedding=embedding,
        dropout_value=0.0,
        aggregation_type='sum'
    )

    model = SimpleClassifier(
        encoder=encoder,
        encoding_dim=EMBEDDING_DIMENSION,
        num_classes=NUM_CLASSES,
        classification_layer_bias=True
    )

    optimizer = optim.Adam(params=model.parameters(),
                           lr=LEARNING_RATE)

    engine = Engine(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        save_dir=MODEL_SAVE_DIR,
        num_epochs=NUM_EPOCHS,
        save_every=SAVE_EVERY
    )

    engine.run()
