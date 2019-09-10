#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bow_bert.py"


python ${SCRIPT_FILE} \
--exp_name "genericsect_bow_bert" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--device cpu \
--bert_type "bert-large-cased" \
--max_length 10 \
--max_num_words 15000 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50
