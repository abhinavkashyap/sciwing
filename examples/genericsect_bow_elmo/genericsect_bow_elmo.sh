#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bow_elmo.py"


python ${SCRIPT_FILE} \
--exp_name "genericsect_bow_elmo" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--max_length 15 \
--max_num_words 15000 \
--device cpu \
--debug \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 1 \
--save_every 5 \
--log_train_metrics_every 10

