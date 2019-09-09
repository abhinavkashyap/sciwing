#!/usr/bin/env bash

SCRIPT_FILE="sectlabel_bow_random.py"

python ${SCRIPT_FILE} \
--exp_name "sectlabel_bow_random" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--max_num_words 3000 \
--max_length 15 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 50 \
--lr 1e-4 \
--epochs 15 \
--save_every 1 \
--log_train_metrics_every 50