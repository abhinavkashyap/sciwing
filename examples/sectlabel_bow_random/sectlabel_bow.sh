#!/usr/bin/env bash

SCRIPT_FILE="sectlabel_bow.py"

python ${SCRIPT_FILE} \
--exp_name "sectlabel_bow_random" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--bs 32 \
--emb_type glove_6B_50 \
--lr 1e-4 \
--epochs 15 \
--save_every 1 \
--log_train_metrics_every 50