from parsect.models.simpleclassifier import SimpleClassifier
from parsect.datasets.parsect_dataset import ParsectDataset
from parsect.modules.bow_encoder import BOW_Encoder
import parsect.constants as constants
import os
import torch.nn as nn
import torch.optim as optim
from parsect.engine.engine import Engine
import json

FILES = constants.FILES
PATHS = constants.PATHS

SECT_LABEL_FILE = FILES['SECT_LABEL_FILE']
OUTPUT_DIR = PATHS['OUTPUT_DIR']
CONFIGS_DIR = PATHS['CONFIGS_DIR']

if __name__ == '__main__':
    # read the hyperparams from config file
    with open(os.path.join(CONFIGS_DIR, 'bow_random_emb_linear_classifier_settings.json')) as fp:
        config = json.load(fp)

    EXP_NAME = config['EXP_NAME']
    EXP_DIR_PATH = os.path.join(OUTPUT_DIR, EXP_NAME)
    MODEL_SAVE_DIR = os.path.join(EXP_DIR_PATH, 'checkpoints')
    if not os.path.isdir(EXP_DIR_PATH):
        os.mkdir(EXP_DIR_PATH)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    MAX_NUM_WORDS = config['MAX_NUM_WORDS']
    MAX_LENGTH = config['MAX_LENGTH']
    VOCAB_STORE_LOCATION = os.path.join(EXP_DIR_PATH, 'vocab.json')
    DEBUG = config['DEBUG']
    DEBUG_DATASET_PROPORTION = config['DEBUG_DATASET_PROPORTION']
    BATCH_SIZE = config['BATCH_SIZE']
    EMBEDDING_DIMENSION = config['EMBEDDING_DIMENSION']
    LEARNING_RATE = config['LEARNING_RATE']
    NUM_EPOCHS = config['NUM_EPOCHS']
    SAVE_EVERY = config['SAVE_EVERY']

    train_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='train',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION
    )

    validation_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='valid',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
        debug=DEBUG,
        debug_dataset_proportion=DEBUG_DATASET_PROPORTION
    )

    test_dataset = ParsectDataset(
        secthead_label_file=SECT_LABEL_FILE,
        dataset_type='test',
        max_num_words=MAX_NUM_WORDS,
        max_length=MAX_LENGTH,
        vocab_store_location=VOCAB_STORE_LOCATION,
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
        save_every=SAVE_EVERY,
    )

    engine.run()

    config['VOCAB_STORE_LOCATION'] = VOCAB_STORE_LOCATION
    config['MODEL_SAVE_DIR'] = MODEL_SAVE_DIR
    config['VOCAB_SIZE'] = VOCAB_SIZE
    config['NUM_CLASSES'] = NUM_CLASSES
    with open(os.path.join(EXP_DIR_PATH, 'config.json'), 'w') as fp:
        json.dump(config, fp)
